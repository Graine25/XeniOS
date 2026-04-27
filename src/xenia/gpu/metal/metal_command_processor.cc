/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/metal/metal_command_processor.h"

#include <dispatch/dispatch.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "third_party/metal-cpp/Foundation/NSURL.hpp"
#include "third_party/metal-cpp/Metal/MTLEvent.hpp"

#include "third_party/fmt/include/fmt/format.h"
#include "xenia/base/assert.h"
#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/base/math.h"
#include "xenia/base/profiling.h"
#include "xenia/base/xxhash.h"
#include "xenia/gpu/draw_util.h"
#include "xenia/gpu/gpu_flags.h"
#include "xenia/gpu/graphics_system.h"
#include "xenia/gpu/metal/metal_graphics_system.h"
#include "xenia/gpu/packet_disassembler.h"
#include "xenia/gpu/registers.h"
#include "xenia/gpu/texture_util.h"
#include "xenia/gpu/xenos.h"
#include "xenia/kernel/kernel_state.h"
#include "xenia/kernel/user_module.h"
#include "xenia/ui/metal/metal_presenter.h"

// Metal IR Converter Runtime - defines IRDescriptorTableEntry and bind points
#define IR_RUNTIME_METALCPP
#include "third_party/metal-shader-converter/include/metal_irconverter_runtime.h"

#ifndef DISPATCH_DATA_DESTRUCTOR_NONE
#define DISPATCH_DATA_DESTRUCTOR_NONE DISPATCH_DATA_DESTRUCTOR_DEFAULT
#endif

DECLARE_bool(async_shader_compilation);
DECLARE_bool(clear_memory_page_state);
DECLARE_bool(submit_on_primary_buffer_end);

namespace xe {
namespace gpu {
namespace metal {

namespace {
void GetBoundRenderTargetSize(const MetalRenderTargetCache* render_target_cache,
                              uint32_t fallback_width, uint32_t fallback_height,
                              uint32_t& width_out, uint32_t& height_out) {
  width_out = std::max(fallback_width, uint32_t(1));
  height_out = std::max(fallback_height, uint32_t(1));
  if (!render_target_cache) {
    return;
  }
  MTL::Texture* pass_size_texture = render_target_cache->GetColorTarget(0);
  if (!pass_size_texture) {
    pass_size_texture = render_target_cache->GetDepthTarget();
  }
  if (!pass_size_texture) {
    pass_size_texture = render_target_cache->GetDummyColorTarget();
  }
  if (!pass_size_texture) {
    return;
  }
  width_out =
      std::max(static_cast<uint32_t>(pass_size_texture->width()), uint32_t(1));
  height_out =
      std::max(static_cast<uint32_t>(pass_size_texture->height()), uint32_t(1));
}

void ClampScissorToBounds(draw_util::Scissor& scissor, uint32_t width,
                          uint32_t height) {
  width = std::max(width, uint32_t(1));
  height = std::max(height, uint32_t(1));

  scissor.offset[0] = std::min(scissor.offset[0], width);
  scissor.offset[1] = std::min(scissor.offset[1], height);

  uint32_t max_scissor_width = width - scissor.offset[0];
  uint32_t max_scissor_height = height - scissor.offset[1];
  scissor.extent[0] = std::min(scissor.extent[0], max_scissor_width);
  scissor.extent[1] = std::min(scissor.extent[1], max_scissor_height);
}

constexpr size_t kResolvedMemoryRangesMax = 8192;
PipelineAttachmentFormats ResolvePipelineAttachmentFormats(
    const MetalRenderTargetCache* render_target_cache,
    MTL::RenderPassDescriptor* pass_descriptor, bool pixel_shader_writes_depth,
    const char* pipeline_name) {
  PipelineAttachmentFormats result;
  result.sample_count = 1;
  for (uint32_t i = 0; i < 4; ++i) {
    result.color_formats[i] = MTL::PixelFormatInvalid;
  }
  result.depth_format = MTL::PixelFormatInvalid;
  result.stencil_format = MTL::PixelFormatInvalid;

  if (render_target_cache) {
    for (uint32_t i = 0; i < 4; ++i) {
      if (MTL::Texture* rt = render_target_cache->GetColorTargetForDraw(i)) {
        result.color_formats[i] = rt->pixelFormat();
        if (rt->sampleCount() > 0) {
          result.sample_count = std::max<uint32_t>(
              result.sample_count, static_cast<uint32_t>(rt->sampleCount()));
        }
      }
    }
    if (result.color_formats[0] == MTL::PixelFormatInvalid) {
      if (MTL::Texture* dummy =
              render_target_cache->GetDummyColorTargetForDraw()) {
        result.color_formats[0] = dummy->pixelFormat();
        if (dummy->sampleCount() > 0) {
          result.sample_count = std::max<uint32_t>(
              result.sample_count, static_cast<uint32_t>(dummy->sampleCount()));
        }
      }
    }
    if (MTL::Texture* depth_tex =
            render_target_cache->GetDepthTargetForDraw()) {
      result.depth_format = depth_tex->pixelFormat();
      switch (result.depth_format) {
        case MTL::PixelFormatDepth32Float_Stencil8:
        case MTL::PixelFormatDepth24Unorm_Stencil8:
        case MTL::PixelFormatX32_Stencil8:
          result.stencil_format = result.depth_format;
          break;
        default:
          result.stencil_format = MTL::PixelFormatInvalid;
          break;
      }
      if (depth_tex->sampleCount() > 0) {
        result.sample_count =
            std::max<uint32_t>(result.sample_count,
                               static_cast<uint32_t>(depth_tex->sampleCount()));
      }
    }
  }

  if (pass_descriptor) {
    // Rebuild strictly from the active encoder descriptor.
    result.sample_count = 1;
    for (uint32_t i = 0; i < 4; ++i) {
      result.color_formats[i] = MTL::PixelFormatInvalid;
    }
    result.depth_format = MTL::PixelFormatInvalid;
    result.stencil_format = MTL::PixelFormatInvalid;

    auto update_sample_count = [&](MTL::Texture* texture) {
      if (!texture) return;
      NS::UInteger sc = texture->sampleCount();
      if (sc > 0) {
        result.sample_count =
            std::max<uint32_t>(result.sample_count, static_cast<uint32_t>(sc));
      }
    };

    auto* color_attachments = pass_descriptor->colorAttachments();
    for (uint32_t i = 0; i < 4; ++i) {
      auto* attachment =
          color_attachments ? color_attachments->object(i) : nullptr;
      if (!attachment) continue;
      MTL::Texture* texture = attachment->texture();
      if (!texture) continue;
      result.color_formats[i] = texture->pixelFormat();
      update_sample_count(texture);
    }
    if (auto* depth_attachment = pass_descriptor->depthAttachment()) {
      MTL::Texture* texture = depth_attachment->texture();
      if (texture) {
        result.depth_format = texture->pixelFormat();
        update_sample_count(texture);
      }
    }
    if (auto* stencil_attachment = pass_descriptor->stencilAttachment()) {
      MTL::Texture* texture = stencil_attachment->texture();
      if (texture) {
        result.stencil_format = texture->pixelFormat();
        update_sample_count(texture);
      }
    }
    // Propagate combined depth-stencil formats.
    if (result.depth_format != MTL::PixelFormatInvalid &&
        result.stencil_format == MTL::PixelFormatInvalid) {
      switch (result.depth_format) {
        case MTL::PixelFormatDepth32Float_Stencil8:
        case MTL::PixelFormatDepth24Unorm_Stencil8:
        case MTL::PixelFormatX32_Stencil8:
          result.stencil_format = result.depth_format;
          break;
        default:
          break;
      }
    } else if (result.stencil_format != MTL::PixelFormatInvalid &&
               result.depth_format == MTL::PixelFormatInvalid) {
      switch (result.stencil_format) {
        case MTL::PixelFormatDepth32Float_Stencil8:
        case MTL::PixelFormatDepth24Unorm_Stencil8:
        case MTL::PixelFormatX32_Stencil8:
          result.depth_format = result.stencil_format;
          break;
        default:
          break;
      }
    }
  }

  // Ensure depth format for depth-writing fragment shaders.
  if (pixel_shader_writes_depth &&
      result.depth_format == MTL::PixelFormatInvalid) {
    result.depth_format = MTL::PixelFormatDepth32Float;
    static bool logged = false;
    if (!logged) {
      logged = true;
      XELOGW(
          "{}: fragment writes depth without a bound depth attachment; "
          "using Depth32Float pipeline fallback",
          pipeline_name);
    }
  }

  return result;
}

MTL::ComputePipelineState* CreateComputePipelineFromEmbeddedLibrary(
    MTL::Device* device, const void* metallib_data, size_t metallib_size,
    const char* debug_name) {
  if (!device || !metallib_data || !metallib_size) {
    return nullptr;
  }

  NS::Error* error = nullptr;
  dispatch_data_t data = dispatch_data_create(
      metallib_data, metallib_size, nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
  MTL::Library* lib = device->newLibrary(data, &error);
  dispatch_release(data);
  if (!lib) {
    XELOGE("Metal: failed to create {} library: {}", debug_name,
           error ? error->localizedDescription()->utf8String() : "unknown");
    return nullptr;
  }

  // XeSL compute entrypoint name used in the embedded metallibs.
  NS::String* fn_name = NS::String::string("entry_xe", NS::UTF8StringEncoding);
  MTL::Function* fn = lib->newFunction(fn_name);
  if (!fn) {
    XELOGE("Metal: {} missing entry_xe", debug_name);
    lib->release();
    return nullptr;
  }

  MTL::ComputePipelineState* pipeline =
      device->newComputePipelineState(fn, &error);
  fn->release();
  lib->release();

  if (!pipeline) {
    XELOGE("Metal: failed to create {} pipeline: {}", debug_name,
           error ? error->localizedDescription()->utf8String() : "unknown");
    return nullptr;
  }

  return pipeline;
}

bool ShaderUsesVertexFetch(const Shader& shader) {
  if (!shader.vertex_bindings().empty()) {
    return true;
  }
  const Shader::ConstantRegisterMap& constant_map =
      shader.constant_register_map();
  for (uint32_t i = 0; i < xe::countof(constant_map.vertex_fetch_bitmap); ++i) {
    if (constant_map.vertex_fetch_bitmap[i] != 0) {
      return true;
    }
  }
  return false;
}

MTL::CompareFunction ToMetalCompareFunction(xenos::CompareFunction compare) {
  static const MTL::CompareFunction kCompareMap[8] = {
      MTL::CompareFunctionNever,         // 0
      MTL::CompareFunctionLess,          // 1
      MTL::CompareFunctionEqual,         // 2
      MTL::CompareFunctionLessEqual,     // 3
      MTL::CompareFunctionGreater,       // 4
      MTL::CompareFunctionNotEqual,      // 5
      MTL::CompareFunctionGreaterEqual,  // 6
      MTL::CompareFunctionAlways,        // 7
  };
  return kCompareMap[uint32_t(compare) & 0x7];
}

MTL::StencilOperation ToMetalStencilOperation(xenos::StencilOp op) {
  static const MTL::StencilOperation kStencilOpMap[8] = {
      MTL::StencilOperationKeep,            // 0
      MTL::StencilOperationZero,            // 1
      MTL::StencilOperationReplace,         // 2
      MTL::StencilOperationIncrementClamp,  // 3
      MTL::StencilOperationDecrementClamp,  // 4
      MTL::StencilOperationInvert,          // 5
      MTL::StencilOperationIncrementWrap,   // 6
      MTL::StencilOperationDecrementWrap,   // 7
  };
  return kStencilOpMap[uint32_t(op) & 0x7];
}

}  // namespace

MetalCommandProcessor::MetalCommandProcessor(
    MetalGraphicsSystem* graphics_system, kernel::KernelState* kernel_state)
    : CommandProcessor(graphics_system, kernel_state) {}

MetalCommandProcessor::~MetalCommandProcessor() {
  // End any active render encoder before releasing
  // Note: Only call endEncoding if the encoder is still active
  // (not already ended by a committed command buffer)
  if (current_render_encoder_) {
    // The encoder may already be ended if the command buffer was committed
    // In that case, just release it
    current_render_encoder_->release();
    current_render_encoder_ = nullptr;
  }
  if (current_command_buffer_) {
    current_command_buffer_->release();
    current_command_buffer_ = nullptr;
  }
  WaitForPendingCompletionHandlers();

  // Release pipeline cache (owns shaders, pipelines, shader translation).
  pipeline_cache_.reset();

  for (auto& pair : depth_stencil_state_cache_) {
    if (pair.second) {
      pair.second->release();
    }
  }
  depth_stencil_state_cache_.clear();

  // Release IR Converter runtime buffers and resources
  if (null_buffer_) {
    null_buffer_->release();
    null_buffer_ = nullptr;
  }
  if (null_texture_) {
    null_texture_->release();
    null_texture_ = nullptr;
  }
  if (null_sampler_) {
    null_sampler_->release();
    null_sampler_ = nullptr;
  }
  current_bindless_table_valid_ = false;
  current_bindless_top_level_buffer_ = nullptr;
  current_bindless_top_level_offset_ = 0;
  current_bindless_top_level_gpu_address_ = 0;
  current_bindless_cbv_buffer_ = nullptr;
  current_bindless_cbv_offset_ = 0;
  current_bindless_cbv_gpu_address_ = 0;
}

void MetalCommandProcessor::TracePlaybackWroteMemory(uint32_t base_ptr,
                                                     uint32_t length) {
  if (shared_memory_) {
    shared_memory_->MemoryInvalidationCallback(base_ptr, length, true);
  }
  if (primitive_processor_) {
    primitive_processor_->MemoryInvalidationCallback(base_ptr, length, true);
  }
}

void MetalCommandProcessor::RestoreEdramSnapshot(const void* snapshot) {
  // Restore the guest EDRAM snapshot captured in the trace into the Metal
  // render-target cache so that subsequent host render targets created from
  // EDRAM (via LoadTiledData) see the same initial contents as other
  // backends like D3D12.
  if (!snapshot) {
    XELOGW(
        "MetalCommandProcessor::RestoreEdramSnapshot called with null "
        "snapshot");
    return;
  }
  if (!render_target_cache_) {
    XELOGW(
        "MetalCommandProcessor::RestoreEdramSnapshot called before render "
        "target "
        "cache initialization");
    return;
  }
  // Trace playback frame boundary: drop resolve-write tracking from previous
  // frame before restoring a new snapshot.
  trace_resolve_guard_.Clear();
  render_target_cache_->RestoreEdramSnapshot(snapshot);
}

void MetalCommandProcessor::ClearCaches() {
  CommandProcessor::ClearCaches();
  // TODO(wmarti): Add cache_clear_requested_ flag like D3D12 for deferred
  // clearing of pipeline caches, texture caches, etc.
}

void MetalCommandProcessor::InvalidateGpuMemory() {
  if (shared_memory_) {
    shared_memory_->InvalidateAllPages();
  }
}

void MetalCommandProcessor::ClearReadbackBuffers() {
  // TODO(wmarti): Implement readback buffer clearing when memexport readback
  // is added. See D3D12's readback_buffers_ and memexport_readback_buffers_.
}

ui::metal::MetalProvider& MetalCommandProcessor::GetMetalProvider() const {
  return *static_cast<ui::metal::MetalProvider*>(graphics_system_->provider());
}

uint64_t MetalCommandProcessor::GetCurrentSubmission() const {
  return submission_current_ ? submission_current_ : 1;
}

uint64_t MetalCommandProcessor::GetCompletedSubmission() const {
  return completed_command_buffers_.load(std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// TraceResolveGuard -- trace-only resolved-memory tracking.
// ---------------------------------------------------------------------------

void MetalCommandProcessor::TraceResolveGuard::Mark(uint32_t base_ptr,
                                                    uint32_t length) {
  if (length == 0) {
    return;
  }
  constexpr uint64_t kAddressLimit =
      uint64_t(std::numeric_limits<uint32_t>::max()) + 1ull;
  uint64_t merged_base = base_ptr;
  uint64_t merged_end =
      std::min<uint64_t>(merged_base + uint64_t(length), kAddressLimit);
  if (merged_end <= merged_base) {
    return;
  }

  for (size_t i = 0; i < ranges_.size();) {
    const auto& range = ranges_[i];
    const uint64_t range_base = range.base;
    const uint64_t range_end =
        std::min<uint64_t>(range_base + uint64_t(range.length), kAddressLimit);
    // Merge overlapping or adjacent ranges.
    if (merged_end + 1 < range_base || range_end + 1 < merged_base) {
      ++i;
      continue;
    }
    merged_base = std::min(merged_base, range_base);
    merged_end = std::max(merged_end, range_end);
    ranges_.erase(ranges_.begin() + i);
  }

  const uint64_t merged_length_64 =
      std::min<uint64_t>(merged_end - merged_base, kAddressLimit - merged_base);
  if (!merged_length_64) {
    return;
  }
  ResolvedRange merged_range = {uint32_t(merged_base),
                                uint32_t(merged_length_64)};
  auto insert_it =
      std::lower_bound(ranges_.begin(), ranges_.end(), merged_range,
                       [](const ResolvedRange& lhs, const ResolvedRange& rhs) {
                         return lhs.base < rhs.base;
                       });
  ranges_.insert(insert_it, merged_range);

  if (ranges_.size() <= kResolvedMemoryRangesMax) {
    return;
  }

  std::sort(ranges_.begin(), ranges_.end(),
            [](const ResolvedRange& lhs, const ResolvedRange& rhs) {
              return lhs.base < rhs.base;
            });
  while (ranges_.size() > kResolvedMemoryRangesMax) {
    size_t best_index = std::numeric_limits<size_t>::max();
    uint64_t best_gap = std::numeric_limits<uint64_t>::max();
    for (size_t i = 0; i + 1 < ranges_.size(); ++i) {
      const auto& left = ranges_[i];
      const auto& right = ranges_[i + 1];
      const uint64_t left_end = uint64_t(left.base) + uint64_t(left.length);
      const uint64_t right_base = uint64_t(right.base);
      const uint64_t gap = right_base > left_end ? right_base - left_end : 0;
      if (gap < best_gap) {
        best_gap = gap;
        best_index = i;
        if (!gap) {
          break;
        }
      }
    }
    if (best_index == std::numeric_limits<size_t>::max()) {
      break;
    }
    auto& left = ranges_[best_index];
    const auto& right = ranges_[best_index + 1];
    const uint64_t merged_base_64 =
        std::min<uint64_t>(left.base, uint64_t(right.base));
    const uint64_t merged_end_64 =
        std::max<uint64_t>(uint64_t(left.base) + uint64_t(left.length),
                           uint64_t(right.base) + uint64_t(right.length));
    const uint64_t merged_len_64 = std::min<uint64_t>(
        merged_end_64 - merged_base_64, kAddressLimit - merged_base_64);
    left.base = uint32_t(merged_base_64);
    left.length = uint32_t(std::max<uint64_t>(1, merged_len_64));
    ranges_.erase(ranges_.begin() + best_index + 1);
  }
}

bool MetalCommandProcessor::TraceResolveGuard::IsResolved(
    uint32_t base_ptr, uint32_t length) const {
  const uint64_t end_ptr = uint64_t(base_ptr) + uint64_t(length);
  for (const auto& range : ranges_) {
    const uint64_t range_end = uint64_t(range.base) + uint64_t(range.length);
    // Check if ranges overlap.
    if (uint64_t(base_ptr) < range_end && end_ptr > uint64_t(range.base)) {
      return true;
    }
  }
  return false;
}

void MetalCommandProcessor::TraceResolveGuard::Clear() { ranges_.clear(); }

void MetalCommandProcessor::ForceIssueSwap() {
  // Force a swap to push any pending render target to presenter
  // This is used by trace dumps to capture output when there's no explicit swap
  if (saw_swap_) {
    return;
  }
  IssueSwap(0, 1280, 720);
}

void MetalCommandProcessor::SetSwapDestSwap(uint32_t dest_base, bool swap) {
  if (!dest_base) {
    return;
  }
  if (swap_dest_swaps_by_base_.size() > 256) {
    swap_dest_swaps_by_base_.clear();
  }
  swap_dest_swaps_by_base_[dest_base] = swap;
}

bool MetalCommandProcessor::ConsumeSwapDestSwap(uint32_t dest_base,
                                                bool* swap_out) {
  if (!swap_out || !dest_base) {
    return false;
  }
  auto it = swap_dest_swaps_by_base_.find(dest_base);
  if (it == swap_dest_swaps_by_base_.end()) {
    return false;
  }
  *swap_out = it->second;
  swap_dest_swaps_by_base_.erase(it);
  return true;
}

bool MetalCommandProcessor::SetupContext() {
  saw_swap_ = false;
  last_swap_ptr_ = 0;
  last_swap_width_ = 0;
  last_swap_height_ = 0;
  swap_dest_swaps_by_base_.clear();
  gamma_ramp_256_entry_table_up_to_date_ = false;
  gamma_ramp_pwl_up_to_date_ = false;

  if (!CommandProcessor::SetupContext()) {
    XELOGE("Failed to initialize base command processor context");
    return false;
  }

  const ui::metal::MetalProvider& provider = GetMetalProvider();
  device_ = provider.GetDevice();
  command_queue_ = provider.GetCommandQueue();

  if (!device_ || !command_queue_) {
    XELOGE("MetalCommandProcessor: No Metal device or command queue available");
    return false;
  }

  wait_shared_event_ = device_->newSharedEvent();
  if (wait_shared_event_) {
    wait_shared_event_->setLabel(
        NS::String::string("XeniaWaitEvent", NS::UTF8StringEncoding));
    wait_shared_event_value_ = 0;
  } else {
    XELOGW(
        "MetalCommandProcessor: SharedEvent unavailable; falling back to "
        "waitUntilCompleted");
  }

  bool supports_apple7 = device_->supportsFamily(MTL::GPUFamilyApple7);
  bool supports_mac2 = device_->supportsFamily(MTL::GPUFamilyMac2);
  mesh_shader_supported_ = supports_apple7 || supports_mac2;

  // Initialize shared memory
  shared_memory_ = std::make_unique<MetalSharedMemory>(*this, *memory_);
  if (!shared_memory_->Initialize()) {
    XELOGE("Failed to initialize shared memory");
    return false;
  }

  // Initialize primitive processor (index/primitive conversion like D3D12).
  primitive_processor_ = std::make_unique<MetalPrimitiveProcessor>(
      *this, *register_file_, *memory_, trace_writer_, *shared_memory_);
  if (!primitive_processor_->Initialize()) {
    XELOGE("Failed to initialize Metal primitive processor");
    return false;
  }

  // Create persistent bindless descriptor heaps BEFORE the texture/sampler
  // caches, because their Initialize() allocates persistent heap slots for
  // null textures and samplers.
  view_bindless_heap_ = device_->newBuffer(
      kViewBindlessHeapSize * sizeof(IRDescriptorTableEntry),
      MTL::ResourceStorageModeShared | MTL::ResourceCPUCacheModeWriteCombined);
  if (!view_bindless_heap_) {
    XELOGE("Failed to create view bindless heap");
    return false;
  }
  view_bindless_heap_->setLabel(
      NS::String::string("XeniaViewBindlessHeap", NS::UTF8StringEncoding));
  memset(view_bindless_heap_->contents(), 0, view_bindless_heap_->length());

  sampler_bindless_heap_ = device_->newBuffer(
      kSamplerBindlessHeapSize * sizeof(IRDescriptorTableEntry),
      MTL::ResourceStorageModeShared | MTL::ResourceCPUCacheModeWriteCombined);
  if (!sampler_bindless_heap_) {
    XELOGE("Failed to create sampler bindless heap");
    return false;
  }
  sampler_bindless_heap_->setLabel(
      NS::String::string("XeniaSamplerBindlessHeap", NS::UTF8StringEncoding));
  memset(sampler_bindless_heap_->contents(), 0,
         sampler_bindless_heap_->length());

  // The bindless heaps are dedicated to dynamically indexed textures and
  // samplers only. System descriptors such as shared memory / EDRAM use small
  // explicit-layout top-level tables populated below once all resources exist.
  view_bindless_heap_next_ = 0;
  view_bindless_heap_exhausted_logged_ = false;
  sampler_bindless_heap_next_ = 0;
  sampler_bindless_heap_exhausted_logged_ = false;

  texture_cache_ = std::make_unique<MetalTextureCache>(this, *register_file_,
                                                       *shared_memory_, 1, 1);
  if (!texture_cache_->Initialize()) {
    XELOGE("Failed to initialize Metal texture cache");
    return false;
  }

  // Initialize render target cache
  render_target_cache_ = std::make_unique<MetalRenderTargetCache>(
      *register_file_, *memory_, &trace_writer_, 1, 1, *this);
  if (!render_target_cache_->Initialize()) {
    XELOGE("Failed to initialize Metal render target cache");
    return false;
  }

  // Create and initialize pipeline cache (shader translation + pipeline
  // management).
  pipeline_cache_ =
      std::make_unique<MetalPipelineCache>(device_, *register_file_);
  {
    bool edram_rov_used = false;
    bool gamma_render_target_as_unorm8 =
        !(edram_rov_used ||
          render_target_cache_->gamma_render_target_as_unorm16());
    if (!pipeline_cache_->InitializeShaderTranslation(
            gamma_render_target_as_unorm8,
            render_target_cache_->msaa_2x_supported(),
            render_target_cache_->draw_resolution_scale_x(),
            render_target_cache_->draw_resolution_scale_y())) {
      XELOGE("Failed to initialize shader translation");
      return false;
    }
  }
  if (mesh_shader_supported_) {
    uint64_t tess_tables_size = IRRuntimeTessellatorTablesSize();
    tessellator_tables_buffer_ =
        device_->newBuffer(tess_tables_size, MTL::ResourceStorageModeShared);
    if (!tessellator_tables_buffer_) {
      XELOGE("Failed to allocate tessellator tables buffer ({} bytes)",
             tess_tables_size);
      return false;
    }
    tessellator_tables_buffer_->setLabel(
        NS::String::string("XeniaTessellatorTables", NS::UTF8StringEncoding));
    IRRuntimeLoadTessellatorTables(tessellator_tables_buffer_);
  }

  // Create the upload buffer pool for per-draw constant buffer allocations.
  constant_buffer_pool_ = std::make_unique<MetalUploadBufferPool>(device_);
  render_encoder_resource_usage_.reserve(128);
  render_encoder_heap_usage_.reserve(32);

  // Create a null buffer for unused descriptor entries
  // This prevents shader validation errors when accessing unpopulated
  // descriptors
  null_buffer_ =
      device_->newBuffer(kNullBufferSize, MTL::ResourceStorageModeShared);
  if (!null_buffer_) {
    XELOGE("Failed to create null buffer");
    return false;
  }
  null_buffer_->setLabel(
      NS::String::string("NullBuffer", NS::UTF8StringEncoding));
  std::memset(null_buffer_->contents(), 0, kNullBufferSize);

  // Create a 1x1x1 placeholder 2D array texture for unbound texture slots
  // Xbox 360 textures are typically 2D arrays (for texture atlases, cubemaps)
  // Using 2DArray prevents "Invalid texture type" validation errors
  MTL::TextureDescriptor* null_tex_desc =
      MTL::TextureDescriptor::alloc()->init();
  null_tex_desc->setTextureType(MTL::TextureType2DArray);
  null_tex_desc->setPixelFormat(MTL::PixelFormatRGBA8Unorm);
  null_tex_desc->setWidth(1);
  null_tex_desc->setHeight(1);
  null_tex_desc->setArrayLength(1);  // Single slice in the array
  null_tex_desc->setStorageMode(MTL::StorageModeShared);
  null_tex_desc->setUsage(MTL::TextureUsageShaderRead);

  null_texture_ = device_->newTexture(null_tex_desc);
  null_tex_desc->release();

  if (!null_texture_) {
    XELOGE("Failed to create null texture");
    return false;
  }
  null_texture_->setLabel(
      NS::String::string("NullTexture2DArray", NS::UTF8StringEncoding));

  // Fill the 1x1x1 texture with opaque white (helps debug if sampled)
  uint32_t white_pixel = 0xFFFFFFFF;
  MTL::Region region =
      MTL::Region(0, 0, 0, 1, 1, 1);  // x,y,z origin, w,h,d size
  null_texture_->replaceRegion(region, 0, 0, &white_pixel, 4, 0);  // slice 0

  // Create a default sampler for unbound sampler slots
  // Must set supportsArgumentBuffers=YES for use in argument buffers
  MTL::SamplerDescriptor* null_smp_desc =
      MTL::SamplerDescriptor::alloc()->init();
  null_smp_desc->setMinFilter(MTL::SamplerMinMagFilterLinear);
  null_smp_desc->setMagFilter(MTL::SamplerMinMagFilterLinear);
  null_smp_desc->setMipFilter(MTL::SamplerMipFilterLinear);
  null_smp_desc->setSAddressMode(MTL::SamplerAddressModeClampToEdge);
  null_smp_desc->setTAddressMode(MTL::SamplerAddressModeClampToEdge);
  null_smp_desc->setRAddressMode(MTL::SamplerAddressModeClampToEdge);
  null_smp_desc->setSupportArgumentBuffers(true);

  null_sampler_ = device_->newSamplerState(null_smp_desc);
  null_smp_desc->release();

  if (!null_sampler_) {
    XELOGE("Failed to create null sampler");
    return false;
  }

  system_view_tables_ = device_->newBuffer(
      kSystemViewTableEntryCount * sizeof(IRDescriptorTableEntry),
      MTL::ResourceStorageModeShared | MTL::ResourceCPUCacheModeWriteCombined);
  if (!system_view_tables_) {
    XELOGE("Failed to create system descriptor tables");
    return false;
  }
  system_view_tables_->setLabel(
      NS::String::string("XeniaSystemViewTables", NS::UTF8StringEncoding));
  std::memset(system_view_tables_->contents(), 0,
              system_view_tables_->length());
  {
    auto* system_entries = reinterpret_cast<IRDescriptorTableEntry*>(
        system_view_tables_->contents());
    MTL::Buffer* shared_mem_buffer =
        shared_memory_ ? shared_memory_->GetBuffer() : nullptr;
    const uint64_t shared_memory_metadata =
        shared_mem_buffer ? shared_mem_buffer->length() : kNullBufferSize;
    IRDescriptorTableSetBuffer(&system_entries[kSystemViewTableSRVSharedMemory],
                               shared_mem_buffer
                                   ? shared_mem_buffer->gpuAddress()
                                   : null_buffer_->gpuAddress(),
                               shared_memory_metadata);
    IRDescriptorTableSetBuffer(&system_entries[kSystemViewTableSRVNull],
                               null_buffer_->gpuAddress(), kNullBufferSize);
    IRDescriptorTableSetBuffer(&system_entries[kSystemViewTableUAVNullStart],
                               null_buffer_->gpuAddress(), kNullBufferSize);
    if (!render_target_cache_ ||
        !render_target_cache_->WriteEdramUintPow2BindlessDescriptor(
            &system_entries[kSystemViewTableUAVNullStart + 1], 2)) {
      XELOGE("Failed to encode typed EDRAM UAV system descriptor");
      return false;
    }
    IRDescriptorTableSetBuffer(
        &system_entries[kSystemViewTableUAVSharedMemoryStart],
        shared_mem_buffer ? shared_mem_buffer->gpuAddress()
                          : null_buffer_->gpuAddress(),
        shared_memory_metadata);
    if (!render_target_cache_ ||
        !render_target_cache_->WriteEdramUintPow2BindlessDescriptor(
            &system_entries[kSystemViewTableUAVSharedMemoryStart + 1], 2)) {
      XELOGE("Failed to encode typed EDRAM UAV system descriptor");
      return false;
    }
  }
  return true;
}

void MetalCommandProcessor::FlushCommandBufferAndWait(uint64_t timeout_ns,
                                                      const char* context) {
  // Phase 1: commit the active command buffer and wait for it.
  if (current_command_buffer_) {
    uint64_t wait_value = 0;
    if (wait_shared_event_) {
      wait_value = ++wait_shared_event_value_;
      current_command_buffer_->encodeSignalEvent(wait_shared_event_,
                                                 wait_value);
    }
    current_command_buffer_->commit();
    if (wait_shared_event_) {
      bool signaled =
          wait_shared_event_->waitUntilSignaledValue(wait_value, timeout_ns);
      if (!signaled) {
        XELOGE("{}: GPU timeout (possible GPU hang)", context);
      }
    } else {
      current_command_buffer_->waitUntilCompleted();
    }
    current_command_buffer_->release();
    current_command_buffer_ = nullptr;
    submission_has_draws_ = false;
    copy_resolve_writes_pending_ = false;
  }
  DrainCommandBufferAutoreleasePool();

  // Phase 2: submit a dummy command buffer to ensure ALL previously committed
  // GPU work completes before the caller tears down resources.
  if (command_queue_) {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::CommandBuffer* sync_cmd = command_queue_->commandBuffer();
    if (sync_cmd) {
      uint64_t wait_value = 0;
      if (wait_shared_event_) {
        wait_value = ++wait_shared_event_value_;
        sync_cmd->encodeSignalEvent(wait_shared_event_, wait_value);
      }
      sync_cmd->commit();
      if (wait_shared_event_) {
        bool signaled =
            wait_shared_event_->waitUntilSignaledValue(wait_value, timeout_ns);
        if (!signaled) {
          XELOGE("{}: GPU sync timeout (possible GPU hang)", context);
        }
      } else {
        sync_cmd->waitUntilCompleted();
      }
    }
    pool->release();
  }
}

void MetalCommandProcessor::PrepareForWait() {
  // Flush pending Metal command buffers before entering wait state so that
  // the worker thread's autorelease pool can drain cleanly.
  EndRenderEncoder();
  FlushCommandBufferAndWait(/*timeout_ns=*/5000000000ULL, "PrepareForWait");
  CommandProcessor::PrepareForWait();
}

void MetalCommandProcessor::WaitForPendingCompletionHandlers() {
  constexpr auto kMaxWait = std::chrono::seconds(5);
  const auto wait_start = std::chrono::steady_clock::now();
  while (pending_completion_handlers_.load(std::memory_order_acquire) != 0) {
    if (std::chrono::steady_clock::now() - wait_start >= kMaxWait) {
      XELOGW(
          "MetalCommandProcessor: timed out waiting for {} completion "
          "handler(s) during shutdown",
          pending_completion_handlers_.load(std::memory_order_relaxed));
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

void MetalCommandProcessor::ShutdownContext() {
  // End the render encoder directly (not via EndRenderEncoder — we release
  // the encoder object below after the command buffer completes).
  if (current_render_encoder_) {
    current_render_encoder_->endEncoding();
  }

  FlushCommandBufferAndWait(std::numeric_limits<uint64_t>::max(),
                            "ShutdownContext");
  WaitForPendingCompletionHandlers();

  // Now safe to release encoder and command buffer
  if (current_render_encoder_) {
    current_render_encoder_->release();
    current_render_encoder_ = nullptr;
  }
  if (current_command_buffer_) {
    current_command_buffer_->release();
    current_command_buffer_ = nullptr;
  }
  DrainCommandBufferAutoreleasePool();

  constant_buffer_pool_.reset();

  // Shut down texture cache first so texture/sampler destructors can release
  // their bindless heap slots while the heap buffers are still alive.
  if (texture_cache_) {
    texture_cache_->Shutdown();
    texture_cache_.reset();
  }

  // Release persistent bindless heaps after texture/sampler caches are
  // torn down (destructors write to these heaps during release).
  if (view_bindless_heap_) {
    view_bindless_heap_->release();
    view_bindless_heap_ = nullptr;
  }
  if (sampler_bindless_heap_) {
    sampler_bindless_heap_->release();
    sampler_bindless_heap_ = nullptr;
  }
  if (system_view_tables_) {
    system_view_tables_->release();
    system_view_tables_ = nullptr;
  }
  view_bindless_heap_free_.clear();
  sampler_bindless_heap_free_.clear();
  retired_view_bindless_indices_.clear();
  retired_sampler_bindless_indices_.clear();
  view_bindless_heap_next_ = 0;
  sampler_bindless_heap_next_ = 0;
  view_bindless_heap_exhausted_logged_ = false;
  sampler_bindless_heap_exhausted_logged_ = false;

  if (primitive_processor_) {
    primitive_processor_->Shutdown();
    primitive_processor_.reset();
  }
  if (tessellator_tables_buffer_) {
    tessellator_tables_buffer_->release();
    tessellator_tables_buffer_ = nullptr;
  }
  frame_open_ = false;

  pipeline_cache_.reset();

  shared_memory_.reset();
  if (wait_shared_event_) {
    wait_shared_event_->release();
    wait_shared_event_ = nullptr;
  }

  CommandProcessor::ShutdownContext();
}

uint32_t MetalCommandProcessor::AllocateViewBindlessIndex() {
  if (!view_bindless_heap_free_.empty()) {
    uint32_t idx = view_bindless_heap_free_.back();
    view_bindless_heap_free_.pop_back();
    view_bindless_heap_exhausted_logged_ = false;
    return idx;
  }
  if (view_bindless_heap_next_ >= kViewBindlessHeapSize) {
    // Reclaim retired indices from completed submissions first — this is cheap
    // and may free slots without needing to evict any textures.
    ProcessCompletedSubmissions();
    if (!view_bindless_heap_free_.empty()) {
      uint32_t idx = view_bindless_heap_free_.back();
      view_bindless_heap_free_.pop_back();
      view_bindless_heap_exhausted_logged_ = false;
      return idx;
    }
    // Still full — try releasing least-recently-used cached bindless views.
    if (texture_cache_ && texture_cache_->TrimViewBindlessPressure()) {
      if (!view_bindless_heap_free_.empty()) {
        uint32_t idx = view_bindless_heap_free_.back();
        view_bindless_heap_free_.pop_back();
        view_bindless_heap_exhausted_logged_ = false;
        return idx;
      }
    }
    if (!view_bindless_heap_exhausted_logged_) {
      uint64_t completed =
          completed_command_buffers_.load(std::memory_order_relaxed);
      XELOGE(
          "View bindless heap exhausted ({} allocated, {} free, {} retired, "
          "submissions: current={} completed={})",
          view_bindless_heap_next_, view_bindless_heap_free_.size(),
          retired_view_bindless_indices_.size(), submission_current_,
          completed);
      view_bindless_heap_exhausted_logged_ = true;
    }
    return UINT32_MAX;
  }
  return view_bindless_heap_next_++;
}

void MetalCommandProcessor::ReleaseViewBindlessIndex(uint32_t index) {
  if (index >= kViewBindlessHeapSize || !view_bindless_heap_) {
    return;
  }
  // Match D3D12's persistent texture-descriptor lifetime more closely: by the
  // time a Metal texture reaches destruction through the cache, its last GPU
  // use has already completed, so the bindless view slot can be recycled
  // immediately rather than waiting for the current submission to end.
  FreeViewBindlessIndexNow(index);
}

void MetalCommandProcessor::RetireViewBindlessIndex(uint32_t index) {
  if (index >= kViewBindlessHeapSize || !view_bindless_heap_) {
    return;
  }
  uint64_t retirement_submission = GetBindlessDescriptorRetirementSubmission();
  if (!retirement_submission) {
    FreeViewBindlessIndexNow(index);
  } else {
    retired_view_bindless_indices_.push_back({index, retirement_submission});
  }
}

uint32_t MetalCommandProcessor::GetViewBindlessHeapAvailableCount() const {
  return uint32_t(kViewBindlessHeapSize - view_bindless_heap_next_) +
         uint32_t(view_bindless_heap_free_.size());
}

uint32_t MetalCommandProcessor::AllocateSamplerBindlessIndex() {
  if (!sampler_bindless_heap_free_.empty()) {
    uint32_t idx = sampler_bindless_heap_free_.back();
    sampler_bindless_heap_free_.pop_back();
    sampler_bindless_heap_exhausted_logged_ = false;
    return idx;
  }
  if (sampler_bindless_heap_next_ >= kSamplerBindlessHeapSize) {
    if (!sampler_bindless_heap_exhausted_logged_) {
      XELOGE("Sampler bindless heap exhausted");
      sampler_bindless_heap_exhausted_logged_ = true;
    }
    return UINT32_MAX;
  }
  return sampler_bindless_heap_next_++;
}

void MetalCommandProcessor::ReleaseSamplerBindlessIndex(uint32_t index) {
  if (index >= kSamplerBindlessHeapSize || !sampler_bindless_heap_) {
    return;
  }
  uint64_t retirement_submission = GetBindlessDescriptorRetirementSubmission();
  if (!retirement_submission) {
    FreeSamplerBindlessIndexNow(index);
  } else {
    retired_sampler_bindless_indices_.push_back({index, retirement_submission});
  }
}

uint64_t MetalCommandProcessor::GetBindlessDescriptorRetirementSubmission()
    const {
  if (!submission_current_) {
    return 0;
  }
  if (current_command_buffer_ ||
      completed_command_buffers_.load(std::memory_order_relaxed) <
          submission_current_) {
    return submission_current_;
  }
  return 0;
}

void MetalCommandProcessor::FreeViewBindlessIndexNow(uint32_t index) {
  if (index >= kViewBindlessHeapSize || !view_bindless_heap_) {
    return;
  }
  if (IRDescriptorTableEntry* entry = GetViewBindlessHeapEntry(index)) {
    std::memset(entry, 0, sizeof(IRDescriptorTableEntry));
  }
  view_bindless_heap_free_.push_back(index);
  view_bindless_heap_exhausted_logged_ = false;
}

void MetalCommandProcessor::FreeSamplerBindlessIndexNow(uint32_t index) {
  if (index >= kSamplerBindlessHeapSize || !sampler_bindless_heap_) {
    return;
  }
  if (IRDescriptorTableEntry* entry = GetSamplerBindlessHeapEntry(index)) {
    std::memset(entry, 0, sizeof(IRDescriptorTableEntry));
  }
  sampler_bindless_heap_free_.push_back(index);
  sampler_bindless_heap_exhausted_logged_ = false;
}

IRDescriptorTableEntry* MetalCommandProcessor::GetViewBindlessHeapEntry(
    uint32_t index) {
  if (!view_bindless_heap_) {
    XELOGE("GetViewBindlessHeapEntry: heap is null!");
    return nullptr;
  }
  if (index >= kViewBindlessHeapSize) {
    XELOGE("GetViewBindlessHeapEntry: index {} >= heap size {}", index,
           kViewBindlessHeapSize);
    return nullptr;
  }
  return reinterpret_cast<IRDescriptorTableEntry*>(
             view_bindless_heap_->contents()) +
         index;
}

IRDescriptorTableEntry* MetalCommandProcessor::GetSamplerBindlessHeapEntry(
    uint32_t index) {
  if (!sampler_bindless_heap_) {
    XELOGE("GetSamplerBindlessHeapEntry: heap is null!");
    return nullptr;
  }
  if (index >= kSamplerBindlessHeapSize) {
    XELOGE("GetSamplerBindlessHeapEntry: index {} >= heap size {}", index,
           kSamplerBindlessHeapSize);
    return nullptr;
  }
  return reinterpret_cast<IRDescriptorTableEntry*>(
             sampler_bindless_heap_->contents()) +
         index;
}

void MetalCommandProcessor::InitializeShaderStorage(
    const std::filesystem::path& cache_root, uint32_t title_id, bool blocking,
    std::function<void()> completion_callback) {
  CommandProcessor::InitializeShaderStorage(cache_root, title_id, blocking,
                                            nullptr);
  if (pipeline_cache_) {
    pipeline_cache_->InitializeShaderStorage(cache_root, title_id, blocking);
  }
  if (completion_callback) {
    completion_callback();
  }
}

void MetalCommandProcessor::IssueSwap(uint32_t frontbuffer_ptr,
                                      uint32_t frontbuffer_width,
                                      uint32_t frontbuffer_height) {
  ProcessCompletedSubmissions();
  saw_swap_ = true;
  last_swap_ptr_ = frontbuffer_ptr;
  last_swap_width_ = frontbuffer_width;
  last_swap_height_ = frontbuffer_height;

  // End any active render encoder
  EndRenderEncoder();

  // Submit and wait for command buffer
  if (current_command_buffer_) {
    current_command_buffer_->commit();
    current_command_buffer_->release();
    current_command_buffer_ = nullptr;
    submission_has_draws_ = false;
    copy_resolve_writes_pending_ = false;
  }
  DrainCommandBufferAutoreleasePool();

  if (primitive_processor_ && frame_open_) {
    primitive_processor_->EndFrame();
    frame_open_ = false;
  }
  // Proactive descriptor-pressure trimming: at frame boundaries the GPU has
  // likely completed prior submissions, so retired descriptor indices can be
  // reclaimed and the texture cache can be trimmed before we start the next
  // frame under pressure.  The high-water mark (75% utilization) gives
  // headroom so that mid-frame allocation bursts don't immediately exhaust
  // the heap.
  if (texture_cache_) {
    constexpr uint32_t kDescriptorPressureThreshold =
        kViewBindlessHeapSize / 4;  // trim when < 25% free
    uint32_t available = GetViewBindlessHeapAvailableCount();
    if (available < kDescriptorPressureThreshold) {
      texture_cache_->TrimViewBindlessPressure(kDescriptorPressureThreshold);
    }
  }

  // Frame boundary reached - resolved memory tracking is only needed within a
  // frame when trace playback writes memory.
  trace_resolve_guard_.Clear();
  if (shared_memory_ && ::cvars::clear_memory_page_state) {
    shared_memory_->SetSystemPageBlocksValidWithGpuDataWritten();
  }

  // Push the rendered frame to the presenter's guest output mailbox.
  // This is required for trace dumps to capture the output via the
  // MetalRenderTargetCache color target (like D3D12).
  auto* presenter =
      static_cast<ui::metal::MetalPresenter*>(graphics_system_->presenter());
  if (presenter && render_target_cache_) {
    uint32_t output_width = frontbuffer_width ? frontbuffer_width : 1280;
    uint32_t output_height = frontbuffer_height ? frontbuffer_height : 720;

    MTL::Texture* source_texture = nullptr;
    bool use_pwl_gamma_ramp = false;
    if (texture_cache_) {
      uint32_t swap_width = 0;
      uint32_t swap_height = 0;
      xenos::TextureFormat swap_format = xenos::TextureFormat::k_8_8_8_8;
      source_texture = texture_cache_->RequestSwapTexture(
          swap_width, swap_height, swap_format);
      if (source_texture) {
        output_width = swap_width;
        output_height = swap_height;
        use_pwl_gamma_ramp =
            swap_format == xenos::TextureFormat::k_2_10_10_10 ||
            swap_format == xenos::TextureFormat::k_2_10_10_10_AS_16_16_16_16;
        if (presenter) {
          if (!gamma_ramp_256_entry_table_up_to_date_ ||
              !gamma_ramp_pwl_up_to_date_) {
            constexpr size_t kGammaRampTableBytes =
                sizeof(reg::DC_LUT_30_COLOR) * 256;
            constexpr size_t kGammaRampPwlBytes =
                sizeof(reg::DC_LUT_PWL_DATA) * 128 * 3;
            if (presenter->UpdateGammaRamp(
                    gamma_ramp_256_entry_table(), kGammaRampTableBytes,
                    gamma_ramp_pwl_rgb(), kGammaRampPwlBytes)) {
              gamma_ramp_256_entry_table_up_to_date_ = true;
              gamma_ramp_pwl_up_to_date_ = true;
            } else {
              XELOGW("Metal IssueSwap: gamma ramp upload failed");
            }
          }
        }
      }
    }

    bool swap_dest_swap = false;
    const bool has_swap_dest_swap =
        ConsumeSwapDestSwap(frontbuffer_ptr, &swap_dest_swap);
    bool force_swap_rb = has_swap_dest_swap && swap_dest_swap;

    if (!source_texture) {
      static bool missing_swap_logged = false;
      if (!missing_swap_logged) {
        missing_swap_logged = true;
        XELOGW(
            "MetalCommandProcessor::IssueSwap: swap texture unavailable; "
            "presenting inactive (black) output");
      }
      presenter->RefreshGuestOutput(
          0, 0, 0, 0, [](ui::Presenter::GuestOutputRefreshContext&) -> bool {
            return false;
          });
      return;
    }

    if (source_texture) {
      ui::metal::MetalPresenter* metal_presenter = presenter;
      uint32_t source_width = output_width;
      uint32_t source_height = output_height;
      bool force_swap_rb_copy = force_swap_rb;
      bool use_pwl_gamma_ramp_copy = use_pwl_gamma_ramp;
      auto aspect = graphics_system_->GetScaledAspectRatio();
      presenter->RefreshGuestOutput(
          output_width, output_height, aspect.first, aspect.second,
          [source_texture, metal_presenter, source_width, source_height,
           force_swap_rb_copy, use_pwl_gamma_ramp_copy](
              ui::Presenter::GuestOutputRefreshContext& context) -> bool {
            auto& metal_context =
                static_cast<ui::metal::MetalGuestOutputRefreshContext&>(
                    context);
            context.SetIs8bpc(!use_pwl_gamma_ramp_copy);
            uint64_t submission_id = 0;
            bool copy_success = metal_presenter->CopyTextureToGuestOutput(
                source_texture, metal_context.resource_uav_capable(),
                source_width, source_height, force_swap_rb_copy,
                use_pwl_gamma_ramp_copy, &submission_id);
            if (copy_success && submission_id) {
              metal_context.SetSubmissionId(submission_id);
            }
            return copy_success;
          });
    }
  }
}

void MetalCommandProcessor::OnPrimaryBufferEnd() {
  if (!current_command_buffer_) {
    return;
  }

  if (!cvars::submit_on_primary_buffer_end) {
    return;
  }

  if (!copy_resolve_writes_pending_ && !CanEndSubmissionImmediately()) {
    return;
  }
  EndCommandBuffer();
}

bool MetalCommandProcessor::CanEndSubmissionImmediately() {
  if (!current_command_buffer_) return false;
  if (pipeline_cache_ && pipeline_cache_->IsCreatingPipelines()) return false;
  return true;
}

Shader* MetalCommandProcessor::LoadShader(xenos::ShaderType shader_type,
                                          uint32_t guest_address,
                                          const uint32_t* host_address,
                                          uint32_t dword_count) {
  return pipeline_cache_->LoadShader(shader_type, guest_address, host_address,
                                     dword_count);
}

bool MetalCommandProcessor::IssueDraw(xenos::PrimitiveType primitive_type,
                                      uint32_t index_count,
                                      IndexBufferInfo* index_buffer_info,
                                      bool major_mode_explicit) {
  const RegisterFile& regs = *register_file_;
  uint32_t normalized_color_mask = 0;

  // Check for copy mode
  xenos::EdramMode edram_mode = regs.Get<reg::RB_MODECONTROL>().edram_mode;
  if (edram_mode != xenos::EdramMode::kCopy && copy_resolve_writes_pending_) {
    // Preserve resolve write visibility when transitioning from copy-only
    // bursts to regular draw work.  Metal has no UAV barrier equivalent, so
    // the command-buffer boundary is the visibility guarantee.
    EndCommandBuffer();
  }
  if (edram_mode == xenos::EdramMode::kCopy) {
    return IssueCopy();
  }

  if (regs.Get<reg::RB_SURFACE_INFO>().surface_pitch == 0) {
    // Doesn't actually draw.
    return true;
  }

  // Vertex shader analysis.
  Shader* vertex_shader = active_vertex_shader();
  if (!vertex_shader) {
    XELOGW("IssueDraw: No vertex shader");
    return false;
  }
  if (!vertex_shader->is_ucode_analyzed()) {
    vertex_shader->AnalyzeUcode(pipeline_cache_->ucode_disasm_buffer());
  }
  bool memexport_used_vertex = vertex_shader->memexport_eM_written() != 0;

  // Pixel shader analysis.
  bool primitive_polygonal = draw_util::IsPrimitivePolygonal(regs);
  bool is_rasterization_done =
      draw_util::IsRasterizationPotentiallyDone(regs, primitive_polygonal);
  Shader* pixel_shader = nullptr;
  if (is_rasterization_done) {
    if (edram_mode == xenos::EdramMode::kColorDepth) {
      pixel_shader = active_pixel_shader();
      if (pixel_shader) {
        if (!pixel_shader->is_ucode_analyzed()) {
          pixel_shader->AnalyzeUcode(pipeline_cache_->ucode_disasm_buffer());
        }
        if (!draw_util::IsPixelShaderNeededWithRasterization(*pixel_shader,
                                                             regs)) {
          pixel_shader = nullptr;
        }
      }
    }
  } else {
    if (!memexport_used_vertex) {
      return true;
    }
  }
  bool memexport_used_pixel =
      pixel_shader && (pixel_shader->memexport_eM_written() != 0);
  memexport_ranges_.clear();
  if (memexport_used_vertex) {
    draw_util::AddMemExportRanges(regs, *vertex_shader, memexport_ranges_);
  }
  if (memexport_used_pixel) {
    draw_util::AddMemExportRanges(regs, *pixel_shader, memexport_ranges_);
  }
  // Primitive/index processing (like D3D12/Vulkan).
  PrimitiveProcessor::ProcessingResult primitive_processing_result;
  if (!primitive_processor_) {
    XELOGE("IssueDraw: primitive processor is not initialized");
    return false;
  }
  if (!primitive_processor_->Process(primitive_processing_result)) {
    XELOGE("IssueDraw: primitive processing failed");
    return false;
  }
  if (!primitive_processing_result.host_draw_vertex_count) {
    return true;
  }
  if (primitive_processing_result.host_vertex_shader_type ==
      Shader::HostVertexShaderType::kMemExportCompute) {
    primitive_processing_result.host_vertex_shader_type =
        Shader::HostVertexShaderType::kVertex;
  }

  bool use_tessellation_emulation = false;
  if (primitive_processing_result.IsTessellated()) {
    if (!mesh_shader_supported_) {
      static bool tess_mesh_logged = false;
      if (!tess_mesh_logged) {
        tess_mesh_logged = true;
        XELOGW(
            "Metal: Tessellation emulation requested but mesh shaders are not "
            "supported on this device");
      }
      return true;
    }
    if (!pixel_shader) {
      static bool tess_no_ps_logged = false;
      if (!tess_no_ps_logged) {
        tess_no_ps_logged = true;
        XELOGW(
            "Metal: Tessellation emulation requested without a pixel shader; "
            "using depth-only PS fallback");
      }
    }
    use_tessellation_emulation = true;
  }

  // Configure render targets via MetalRenderTargetCache, similar to D3D12.
  // Update() may internally call PerformTransfersAndResolveClears for EDRAM
  // ownership transfers -- this is the draw-path transfer entry point and
  // is part of the host render backend boundary (see header comment).
  if (render_target_cache_) {
    auto normalized_depth_control = draw_util::GetNormalizedDepthControl(regs);
    uint32_t ps_writes_color_targets =
        pixel_shader ? pixel_shader->writes_color_targets() : 0;
    normalized_color_mask = pixel_shader ? draw_util::GetNormalizedColorMask(
                                               regs, ps_writes_color_targets)
                                         : 0;
    if (!render_target_cache_->Update(is_rasterization_done,
                                      normalized_depth_control,
                                      normalized_color_mask, *vertex_shader)) {
      XELOGE(
          "MetalCommandProcessor::IssueDraw - RenderTargetCache::Update "
          "failed");
      return false;
    }
  }

  // Begin command buffer if needed (will use cache-provided render targets).
  BeginCommandBuffer();
  if (!current_command_buffer_ || !current_render_encoder_) {
    static bool no_command_buffer_logged = false;
    if (!no_command_buffer_logged) {
      no_command_buffer_logged = true;
      XELOGE(
          "IssueDraw: failed to begin Metal command buffer/render encoder; "
          "skipping draws until uniforms buffer allocation recovers");
    }
    return false;
  }

  // =========================================================================
  // MSC (DXBC -> DXIL -> Metal IR) draw path.
  //
  // Guest-facing work (shader analysis, translation, pipeline lookup,
  // shared-memory sync) is performed here.  The host-specific draw
  // backend (UploadConstants / PopulateBindlessTables / DispatchDraw) is
  // invoked at the end of this block.
  // =========================================================================
  // Cast to MSC-specific shader types for the rest of this path.
  auto* metal_vertex_shader = static_cast<MetalShader*>(vertex_shader);
  auto* metal_pixel_shader = static_cast<MetalShader*>(pixel_shader);

  MTL::RenderPipelineState* pipeline = nullptr;
  // Select per-draw shader modifications (mirrors D3D12 PipelineCache).
  uint32_t ps_param_gen_pos = UINT32_MAX;
  uint32_t interpolator_mask = 0;
  if (pixel_shader) {
    interpolator_mask = vertex_shader->writes_interpolators() &
                        pixel_shader->GetInterpolatorInputMask(
                            regs.Get<reg::SQ_PROGRAM_CNTL>(),
                            regs.Get<reg::SQ_CONTEXT_MISC>(), ps_param_gen_pos);
  }

  auto normalized_depth_control = draw_util::GetNormalizedDepthControl(regs);
  Shader::HostVertexShaderType host_vertex_shader_type_for_translation =
      primitive_processing_result.host_vertex_shader_type;
  if (host_vertex_shader_type_for_translation ==
          Shader::HostVertexShaderType::kPointListAsTriangleStrip ||
      host_vertex_shader_type_for_translation ==
          Shader::HostVertexShaderType::kRectangleListAsTriangleStrip) {
    if (!mesh_shader_supported_) {
      static bool host_vs_expansion_logged = false;
      if (!host_vs_expansion_logged) {
        host_vs_expansion_logged = true;
        XELOGW(
            "Metal: Host VS expansion requested without mesh shader support; "
            "skipping draw");
      }
      return true;
    }
    // Geometry emulation handles point/rectangle expansion; use the normal
    // vertex shader translation path to avoid unsupported host VS types.
    host_vertex_shader_type_for_translation =
        Shader::HostVertexShaderType::kVertex;
  }
  DxbcShaderTranslator::Modification vertex_shader_modification =
      pipeline_cache_->GetCurrentVertexShaderModification(
          *vertex_shader, host_vertex_shader_type_for_translation,
          interpolator_mask);
  DxbcShaderTranslator::Modification pixel_shader_modification =
      pixel_shader ? pipeline_cache_->GetCurrentPixelShaderModification(
                         *pixel_shader, interpolator_mask, ps_param_gen_pos,
                         normalized_depth_control)
                   : DxbcShaderTranslator::Modification(0);

  PipelineGeometryShader geometry_shader_type = PipelineGeometryShader::kNone;
  if (!primitive_processing_result.IsTessellated()) {
    switch (primitive_processing_result.host_primitive_type) {
      case xenos::PrimitiveType::kPointList:
        geometry_shader_type = PipelineGeometryShader::kPointList;
        break;
      case xenos::PrimitiveType::kRectangleList:
        geometry_shader_type = PipelineGeometryShader::kRectangleList;
        break;
      case xenos::PrimitiveType::kQuadList:
        geometry_shader_type = PipelineGeometryShader::kQuadList;
        break;
      default:
        break;
    }
  }

  GeometryShaderKey geometry_shader_key;
  bool use_geometry_emulation = false;
  if (geometry_shader_type != PipelineGeometryShader::kNone) {
    bool can_build_geometry_shader =
        pixel_shader || !vertex_shader_modification.vertex.interpolator_mask;
    if (!can_build_geometry_shader) {
      static bool geom_interp_mismatch_logged = false;
      if (!geom_interp_mismatch_logged) {
        geom_interp_mismatch_logged = true;
        XELOGW(
            "Metal: geometry emulation skipped because pixel shader is null "
            "but vertex interpolators are present");
      }
    } else {
      use_geometry_emulation =
          GetGeometryShaderKey(geometry_shader_type, vertex_shader_modification,
                               pixel_shader_modification, geometry_shader_key);
    }
  }
  if (use_geometry_emulation && !mesh_shader_supported_) {
    static bool mesh_support_logged = false;
    if (!mesh_support_logged) {
      mesh_support_logged = true;
      XELOGW(
          "Metal: geometry emulation requested but mesh shaders are not "
          "supported on this device");
    }
    use_geometry_emulation = false;
  }
  if (use_geometry_emulation && !pixel_shader) {
    static bool geom_no_ps_logged = false;
    if (!geom_no_ps_logged) {
      geom_no_ps_logged = true;
      XELOGW(
          "Metal: geometry emulation requested without a pixel shader; using "
          "depth-only PS fallback");
    }
  }

  // Get or create shader translations for the selected modifications.
  auto vertex_translation = static_cast<MetalShader::MetalTranslation*>(
      vertex_shader->GetOrCreateTranslation(vertex_shader_modification.value));
  if (!vertex_translation->is_translated()) {
    if (!pipeline_cache_->shader_translator()->TranslateAnalyzedShader(
            *vertex_translation)) {
      XELOGE("Failed to translate vertex shader to DXBC");
      return false;
    }
  }
  if (!use_tessellation_emulation && !vertex_translation->is_valid()) {
    if (!vertex_translation->TranslateToMetal(
            device_, *pipeline_cache_->dxbc_to_dxil_converter(),
            *pipeline_cache_->metal_shader_converter())) {
      XELOGE("Failed to translate vertex shader to Metal");
      return false;
    }
  }

  MetalShader::MetalTranslation* pixel_translation = nullptr;
  if (pixel_shader) {
    pixel_translation = static_cast<MetalShader::MetalTranslation*>(
        pixel_shader->GetOrCreateTranslation(pixel_shader_modification.value));
    if (!pixel_translation->is_translated()) {
      if (!pipeline_cache_->shader_translator()->TranslateAnalyzedShader(
              *pixel_translation)) {
        XELOGE("Failed to translate pixel shader to DXBC");
        return false;
      }
    }
    if (!pixel_translation->is_valid()) {
      if (!pixel_translation->TranslateToMetal(
              device_, *pipeline_cache_->dxbc_to_dxil_converter(),
              *pipeline_cache_->metal_shader_converter())) {
        XELOGE("Failed to translate pixel shader to Metal");
        return false;
      }
    }
  }

  // Resolve attachment formats once for all pipeline paths.
  bool pixel_shader_writes_depth_for_fmts =
      pixel_translation && pixel_translation->shader().writes_depth();
  if (use_tessellation_emulation && !pixel_translation) {
    pixel_shader_writes_depth_for_fmts = true;  // depth-only PS fallback
  }
  if (use_geometry_emulation && !pixel_translation) {
    pixel_shader_writes_depth_for_fmts = true;  // depth-only PS fallback
  }
  MTL::RenderPassDescriptor* pass_desc_for_fmts =
      current_render_pass_descriptor_;
  if (render_target_cache_) {
    if (MTL::RenderPassDescriptor* cache_desc =
            render_target_cache_->GetRenderPassDescriptor(1)) {
      pass_desc_for_fmts = cache_desc;
    }
  }
  if (!pass_desc_for_fmts) {
    pass_desc_for_fmts = nullptr;
  }
  auto attachment_formats = ResolvePipelineAttachmentFormats(
      render_target_cache_.get(), pass_desc_for_fmts,
      pixel_shader_writes_depth_for_fmts, "Pipeline");

  // Derive the shared rendering key (color mask, blend, alpha-to-mask) once
  // for all pipeline paths instead of re-reading registers in each method.
  bool use_fallback_ps =
      (use_tessellation_emulation || use_geometry_emulation) &&
      !pixel_translation;
  auto rendering_key =
      ResolvePipelineRenderingKey(regs, pixel_translation, use_fallback_ps);

  MetalPipelineCache::TessellationPipelineState* tessellation_pipeline_state =
      nullptr;
  MetalPipelineCache::GeometryPipelineState* geometry_pipeline_state = nullptr;
  if (use_tessellation_emulation) {
    tessellation_pipeline_state =
        pipeline_cache_->GetOrCreateTessellationPipelineState(
            vertex_translation, pixel_translation, primitive_processing_result,
            attachment_formats, rendering_key);
    pipeline = tessellation_pipeline_state
                   ? tessellation_pipeline_state->pipeline
                   : nullptr;
  } else if (use_geometry_emulation) {
    geometry_pipeline_state = pipeline_cache_->GetOrCreateGeometryPipelineState(
        vertex_translation, pixel_translation, geometry_shader_key,
        attachment_formats, rendering_key);
    if (!geometry_pipeline_state || !geometry_pipeline_state->pipeline) {
      static bool geometry_pipeline_failure_logged = false;
      if (!geometry_pipeline_failure_logged) {
        geometry_pipeline_failure_logged = true;
        XELOGW(
            "Metal: geometry emulation pipeline creation failed; skipping "
            "geometry-emulated draws instead of aborting the backend draw "
            "packet");
      }
      return true;
    }
    pipeline = geometry_pipeline_state->pipeline;
  } else {
    auto* pipeline_handle = pipeline_cache_->GetOrCreatePipelineState(
        vertex_translation, pixel_translation, attachment_formats,
        rendering_key);
    if (pipeline_handle) {
      pipeline = pipeline_handle->state.load(std::memory_order_acquire);
    }
    if (!pipeline) {
      if (cvars::async_shader_compilation && pipeline_handle) {
        // Pipeline is being compiled in the background -- skip this draw.
        return true;
      }
      XELOGE("Failed to create pipeline state");
      return false;
    }
  }

  uint32_t used_texture_mask =
      metal_vertex_shader->GetUsedTextureMaskAfterTranslation();
  if (metal_pixel_shader) {
    used_texture_mask |=
        metal_pixel_shader->GetUsedTextureMaskAfterTranslation();
  }
  if (texture_cache_ && used_texture_mask &&
      texture_cache_->AnyUsedTextureRequestWorkPending(used_texture_mask)) {
    texture_cache_->RequestTextures(used_texture_mask);
  }

  std::array<VertexBindingRange, 32> vertex_ranges;
  uint32_t vertex_range_count = 0;
  const auto& vb_bindings = vertex_shader->vertex_bindings();
  bool uses_vertex_fetch = ShaderUsesVertexFetch(*vertex_shader);

  // Sync shared memory before drawing - ensure GPU has latest data
  // This is particularly important for trace playback where memory is
  // written incrementally
  if (shared_memory_) {
    const Shader::ConstantRegisterMap& constant_map_vertex =
        vertex_shader->constant_register_map();
    for (uint32_t i = 0;
         i < xe::countof(constant_map_vertex.vertex_fetch_bitmap); ++i) {
      uint32_t vfetch_bits_remaining =
          constant_map_vertex.vertex_fetch_bitmap[i];
      uint32_t j;
      while (xe::bit_scan_forward(vfetch_bits_remaining, &j)) {
        vfetch_bits_remaining &= ~(uint32_t(1) << j);
        uint32_t vfetch_index = i * 32 + j;
        xenos::xe_gpu_vertex_fetch_t vfetch = regs.GetVertexFetch(vfetch_index);
        switch (vfetch.type) {
          case xenos::FetchConstantType::kVertex:
            break;
          case xenos::FetchConstantType::kInvalidVertex:
            if (::cvars::gpu_allow_invalid_fetch_constants) {
              break;
            }
            XELOGW(
                "Vertex fetch constant {} ({:08X} {:08X}) has \"invalid\" "
                "type. "
                "Use --gpu_allow_invalid_fetch_constants to bypass.",
                vfetch_index, vfetch.dword_0, vfetch.dword_1);
            return false;
          default:
            XELOGW("Vertex fetch constant {} ({:08X} {:08X}) is invalid.",
                   vfetch_index, vfetch.dword_0, vfetch.dword_1);
            return false;
        }
        uint32_t buffer_offset = vfetch.address << 2;
        uint32_t buffer_length = vfetch.size << 2;
        if (buffer_offset > SharedMemory::kBufferSize ||
            SharedMemory::kBufferSize - buffer_offset < buffer_length) {
          XELOGW(
              "Vertex fetch constant {} out of range (offset=0x{:08X} size={})",
              vfetch_index, buffer_offset, buffer_length);
          return false;
        }
        if (!shared_memory_->RequestRange(buffer_offset, buffer_length)) {
          XELOGE(
              "Failed to request vertex buffer at 0x{:08X} (size {}) in shared "
              "memory",
              buffer_offset, buffer_length);
          return false;
        }
      }
    }

    for (const draw_util::MemExportRange& memexport_range : memexport_ranges_) {
      uint32_t base_bytes = memexport_range.base_address_dwords << 2;
      if (!shared_memory_->RequestRange(base_bytes,
                                        memexport_range.size_bytes)) {
        XELOGE(
            "Failed to request memexport stream at 0x{:08X} (size {}) in "
            "shared "
            "memory",
            base_bytes, memexport_range.size_bytes);
        return false;
      }
    }

    for (const auto& binding : vb_bindings) {
      xenos::xe_gpu_vertex_fetch_t vfetch =
          regs.GetVertexFetch(binding.fetch_constant);
      uint32_t buffer_offset = vfetch.address << 2;
      uint32_t buffer_length = vfetch.size << 2;
      VertexBindingRange range;
      range.binding_index = static_cast<uint32_t>(binding.binding_index);
      range.offset = buffer_offset;
      range.length = buffer_length;
      range.stride = binding.stride_words * 4;
      assert_true(vertex_range_count < vertex_ranges.size());
      vertex_ranges[vertex_range_count++] = range;
    }
  }

  if (current_render_pipeline_state_ != pipeline) {
    current_render_encoder_->setRenderPipelineState(pipeline);
    current_render_pipeline_state_ = pipeline;
  }
  if (use_tessellation_emulation) {
    if (!tessellator_tables_buffer_) {
      XELOGE("Tessellation emulation requires tessellator tables buffer");
      return false;
    }
    current_render_encoder_->setObjectBuffer(
        tessellator_tables_buffer_, 0, kIRRuntimeTessellatorTablesBindPoint);
    current_render_encoder_->setMeshBuffer(
        tessellator_tables_buffer_, 0, kIRRuntimeTessellatorTablesBindPoint);
    UseRenderEncoderResource(tessellator_tables_buffer_,
                             MTL::ResourceUsageRead);
  }

  // Determine if shared memory should be UAV (for memexport).
  bool shared_memory_is_uav = memexport_used_vertex || memexport_used_pixel;
  MTL::ResourceUsage shared_memory_usage =
      shared_memory_is_uav ? (MTL::ResourceUsageRead | MTL::ResourceUsageWrite)
                           : MTL::ResourceUsageRead;

  // =========================================================================
  // Host render backend draw entry point.
  //
  // The three virtual methods below form the host draw path.  IssueDraw
  // handles guest-facing validation, shader translation, pipeline lookup,
  // and shared-memory synchronisation above; the host backend is
  // responsible only for uploading constants, populating descriptors, and
  // dispatching the Metal draw call.
  // =========================================================================

  // Upload per-draw constant buffers and apply fixed-function state.
  UniformBufferInfo uniforms;
  if (constant_buffer_pool_ && shared_memory_) {
    if (!UploadConstants(regs, vertex_shader, pixel_shader, metal_vertex_shader,
                         metal_pixel_shader, shared_memory_is_uav,
                         primitive_processing_result, used_texture_mask,
                         normalized_color_mask, uniforms)) {
      return false;
    }

    // Build and bind the per-draw bindless descriptor tables.
    if (!PopulateBindlessTables(metal_vertex_shader, metal_pixel_shader,
                                shared_memory_is_uav, shared_memory_usage,
                                use_geometry_emulation,
                                use_tessellation_emulation, uniforms)) {
      return false;
    }
  }

  // Bind vertex buffers and dispatch the draw.
  bool memexport_used = memexport_used_vertex || memexport_used_pixel;
  return DispatchDraw(regs, primitive_processing_result,
                      use_tessellation_emulation, tessellation_pipeline_state,
                      use_geometry_emulation, geometry_pipeline_state,
                      shared_memory_is_uav, shared_memory_usage, memexport_used,
                      uses_vertex_fetch, vb_bindings, vertex_ranges.data(),
                      vertex_range_count, index_buffer_info);
}

bool MetalCommandProcessor::UploadConstants(
    const RegisterFile& regs, Shader* vertex_shader, Shader* pixel_shader,
    MetalShader* metal_vertex_shader, MetalShader* metal_pixel_shader,
    bool shared_memory_is_uav,
    const PrimitiveProcessor::ProcessingResult& primitive_processing_result,
    uint32_t used_texture_mask, uint32_t normalized_color_mask,
    UniformBufferInfo& uniforms_out) {
  // Determine primitive type characteristics
  bool primitive_polygonal = draw_util::IsPrimitivePolygonal(regs);

  // Get viewport info for NDC transform. Use the actual RT0 dimensions
  // when available so system constants match the current render target.
  uint32_t vp_width = 1;
  uint32_t vp_height = 1;
  GetBoundRenderTargetSize(render_target_cache_.get(), 1280, 720, vp_width,
                           vp_height);
  draw_util::ViewportInfo viewport_info;
  auto depth_control = draw_util::GetNormalizedDepthControl(regs);
  constexpr uint32_t kViewportBoundsMax = 32767;
  bool host_render_targets_used = true;
  bool convert_z_to_float24 = host_render_targets_used &&
                              ::cvars::depth_float24_convert_in_pixel_shader;
  uint32_t draw_resolution_scale_x =
      texture_cache_ ? texture_cache_->draw_resolution_scale_x() : 1;
  uint32_t draw_resolution_scale_y =
      texture_cache_ ? texture_cache_->draw_resolution_scale_y() : 1;
  draw_util::GetViewportInfoArgs gviargs{};
  gviargs.Setup(
      draw_resolution_scale_x, draw_resolution_scale_y,
      texture_cache_ ? texture_cache_->draw_resolution_scale_x_divisor()
                     : divisors::MagicDiv(1),
      texture_cache_ ? texture_cache_->draw_resolution_scale_y_divisor()
                     : divisors::MagicDiv(1),
      true, kViewportBoundsMax, kViewportBoundsMax, false, depth_control,
      convert_z_to_float24, host_render_targets_used,
      pixel_shader && pixel_shader->writes_depth());
  gviargs.SetupRegisterValues(regs);
  draw_util::GetHostViewportInfo(&gviargs, viewport_info);

  // Apply per-draw viewport and scissor so the Metal viewport
  // matches the guest viewport computed by draw_util.
  draw_util::Scissor scissor;
  draw_util::GetScissor(regs, scissor);
  // draw_resolution_scale_x/y already computed above for viewport.
  scissor.offset[0] *= draw_resolution_scale_x;
  scissor.offset[1] *= draw_resolution_scale_y;
  scissor.extent[0] *= draw_resolution_scale_x;
  scissor.extent[1] *= draw_resolution_scale_y;

  // Clamp scissor to actual render target bounds (Metal requires this).
  ClampScissorToBounds(scissor, vp_width, vp_height);

  MTL::Viewport mtl_viewport;
  mtl_viewport.originX = static_cast<double>(viewport_info.xy_offset[0]);
  mtl_viewport.originY = static_cast<double>(viewport_info.xy_offset[1]);
  mtl_viewport.width = static_cast<double>(viewport_info.xy_extent[0]);
  mtl_viewport.height = static_cast<double>(viewport_info.xy_extent[1]);
  mtl_viewport.znear = viewport_info.z_min;
  mtl_viewport.zfar = viewport_info.z_max;
  if (viewport_dirty_ || std::memcmp(&mtl_viewport, &cached_viewport_,
                                     sizeof(MTL::Viewport)) != 0) {
    current_render_encoder_->setViewport(mtl_viewport);
    cached_viewport_ = mtl_viewport;
    viewport_dirty_ = false;
  }

  MTL::ScissorRect mtl_scissor;
  mtl_scissor.x = scissor.offset[0];
  mtl_scissor.y = scissor.offset[1];
  mtl_scissor.width = scissor.extent[0];
  mtl_scissor.height = scissor.extent[1];
  if (scissor_dirty_ || std::memcmp(&mtl_scissor, &cached_scissor_,
                                    sizeof(MTL::ScissorRect)) != 0) {
    current_render_encoder_->setScissorRect(mtl_scissor);
    cached_scissor_ = mtl_scissor;
    scissor_dirty_ = false;
  }

  ApplyRasterizerState(primitive_polygonal);

  // Fixed-function depth/stencil state is not part of the pipeline state in
  // Metal, so update it per draw.
  ApplyDepthStencilState(primitive_polygonal, depth_control);

  // Update full system constants from GPU registers.
  // normalized_color_mask was already computed above for render target update.
  UpdateSystemConstantValues(
      shared_memory_is_uav, primitive_polygonal,
      primitive_processing_result.line_loop_closing_index,
      primitive_processing_result.host_shader_index_endian, viewport_info,
      used_texture_mask, depth_control, normalized_color_mask);

  float blend_constants[] = {
      regs.Get<float>(XE_GPU_REG_RB_BLEND_RED),
      regs.Get<float>(XE_GPU_REG_RB_BLEND_GREEN),
      regs.Get<float>(XE_GPU_REG_RB_BLEND_BLUE),
      regs.Get<float>(XE_GPU_REG_RB_BLEND_ALPHA),
  };
  bool blend_factor_update_needed =
      !ff_blend_factor_valid_ ||
      std::memcmp(ff_blend_factor_, blend_constants, sizeof(float) * 4) != 0;
  if (blend_factor_update_needed) {
    std::memcpy(ff_blend_factor_, blend_constants, sizeof(float) * 4);
    ff_blend_factor_valid_ = true;
    current_render_encoder_->setBlendColor(
        blend_constants[0], blend_constants[1], blend_constants[2],
        blend_constants[3]);
  }

  constexpr size_t kStageVertex = 0;
  constexpr size_t kStagePixel = 1;

  // Uniforms buffer layout (4KB per CBV for alignment):
  //   b0 (offset 0):     System constants (~512 bytes)
  //   b1 (offset 4096):  Float constants (256 float4s = 4KB)
  //   b2 (offset 8192):  Bool/loop constants (~256 bytes)
  //   b3 (offset 12288): Fetch constants (768 bytes)
  //   b4 (offset 16384): Descriptor indices (bindless heap slots)
  const size_t kCBVSize = kCbvSizeBytes;
  constexpr size_t kFloatConstantOffset = 1 * kCbvSizeBytes;
  constexpr size_t kBoolLoopConstantOffset = 2 * kCbvSizeBytes;
  constexpr size_t kFetchConstantOffset = 3 * kCbvSizeBytes;
  constexpr size_t kBoolLoopConstantsSize = (8 + 32) * sizeof(uint32_t);
  const size_t kFetchConstantCount =
      xenos::kTextureFetchConstantCount * 6;  // 192 DWORDs = 768 bytes

  // ---------------------------------------------------------------
  // Allocate per-draw uniform blocks from the upload buffer pool.
  // Each stage (VS, PS) gets one contiguous kUniformsBytesPerTable block.
  // When ALL CBVs are up-to-date we skip allocation entirely and rebind
  // the previous block.
  // ---------------------------------------------------------------

  // Check if float constant layout changed (different shader bound).
  // Matches D3D12 d3d12_command_processor.cc:4910-4943.
  {
    const Shader::ConstantRegisterMap& float_map_vs =
        vertex_shader->constant_register_map();
    for (uint32_t i = 0; i < 4; ++i) {
      if (current_float_constant_map_vertex_[i] !=
          float_map_vs.float_bitmap[i]) {
        current_float_constant_map_vertex_[i] = float_map_vs.float_bitmap[i];
        if (float_map_vs.float_count) {
          cbuffer_binding_float_vertex_.up_to_date = false;
        }
      }
    }
    if (pixel_shader) {
      const Shader::ConstantRegisterMap& float_map_ps =
          pixel_shader->constant_register_map();
      for (uint32_t i = 0; i < 4; ++i) {
        if (current_float_constant_map_pixel_[i] !=
            float_map_ps.float_bitmap[i]) {
          current_float_constant_map_pixel_[i] = float_map_ps.float_bitmap[i];
          if (float_map_ps.float_count) {
            cbuffer_binding_float_pixel_.up_to_date = false;
          }
        }
      }
    } else {
      std::memset(current_float_constant_map_pixel_, 0,
                  sizeof(current_float_constant_map_pixel_));
    }
  }

  const auto& texture_bindings_vertex =
      metal_vertex_shader->GetTextureBindingsAfterTranslation();
  const auto& sampler_bindings_vertex =
      metal_vertex_shader->GetSamplerBindingsAfterTranslation();
  const size_t texture_count_vertex = texture_bindings_vertex.size();
  const size_t sampler_count_vertex = sampler_bindings_vertex.size();
  size_t texture_layout_uid_vertex =
      metal_vertex_shader->GetTextureBindingLayoutUserUID();
  size_t sampler_layout_uid_vertex =
      metal_vertex_shader->GetSamplerBindingLayoutUserUID();
  std::vector<uint32_t> next_texture_bindless_indices_vertex;
  std::vector<uint32_t> next_sampler_bindless_indices_vertex;
  if (sampler_count_vertex) {
    if (current_sampler_layout_uid_vertex_ != sampler_layout_uid_vertex) {
      current_sampler_layout_uid_vertex_ = sampler_layout_uid_vertex;
      cbuffer_binding_descriptor_indices_vertex_up_to_date_ = false;
    }
    current_samplers_vertex_.resize(
        std::max(current_samplers_vertex_.size(), sampler_count_vertex));
    for (size_t i = 0; i < sampler_count_vertex; ++i) {
      auto parameters =
          texture_cache_->GetSamplerParameters(sampler_bindings_vertex[i]);
      if (current_samplers_vertex_[i] != parameters) {
        current_samplers_vertex_[i] = parameters;
        cbuffer_binding_descriptor_indices_vertex_up_to_date_ = false;
      }
    }
  } else if (current_sampler_layout_uid_vertex_ != sampler_layout_uid_vertex) {
    current_sampler_layout_uid_vertex_ = sampler_layout_uid_vertex;
    cbuffer_binding_descriptor_indices_vertex_up_to_date_ = false;
  }
  if (current_texture_layout_uid_vertex_ != texture_layout_uid_vertex &&
      !texture_count_vertex) {
    cbuffer_binding_descriptor_indices_vertex_up_to_date_ = false;
  } else if (texture_count_vertex &&
             cbuffer_binding_descriptor_indices_vertex_up_to_date_ &&
             (current_texture_layout_uid_vertex_ != texture_layout_uid_vertex ||
              !texture_cache_->AreActiveTextureSRVKeysUpToDate(
                  current_texture_srv_keys_vertex_.data(),
                  texture_bindings_vertex.data(), texture_count_vertex))) {
    cbuffer_binding_descriptor_indices_vertex_up_to_date_ = false;
  }
  if (!cbuffer_binding_descriptor_indices_vertex_up_to_date_) {
    next_texture_bindless_indices_vertex.reserve(texture_count_vertex);
    for (const auto& binding : texture_bindings_vertex) {
      next_texture_bindless_indices_vertex.push_back(
          texture_cache_->GetBindlessSRVIndexForBinding(
              binding.fetch_constant, binding.dimension, binding.is_signed));
    }
    next_sampler_bindless_indices_vertex.reserve(sampler_count_vertex);
    for (const auto& binding : sampler_bindings_vertex) {
      next_sampler_bindless_indices_vertex.push_back(
          texture_cache_->GetBindlessSamplerIndexForBinding(binding));
    }
  }

  size_t texture_layout_uid_pixel = 0;
  size_t sampler_layout_uid_pixel = 0;
  const std::vector<DxbcShader::TextureBinding>* texture_bindings_pixel_ptr =
      nullptr;
  std::vector<uint32_t> next_texture_bindless_indices_pixel;
  std::vector<uint32_t> next_sampler_bindless_indices_pixel;
  if (metal_pixel_shader) {
    const auto& texture_bindings_pixel =
        metal_pixel_shader->GetTextureBindingsAfterTranslation();
    const auto& sampler_bindings_pixel =
        metal_pixel_shader->GetSamplerBindingsAfterTranslation();
    const size_t texture_count_pixel = texture_bindings_pixel.size();
    const size_t sampler_count_pixel = sampler_bindings_pixel.size();
    texture_bindings_pixel_ptr = &texture_bindings_pixel;
    texture_layout_uid_pixel =
        metal_pixel_shader->GetTextureBindingLayoutUserUID();
    sampler_layout_uid_pixel =
        metal_pixel_shader->GetSamplerBindingLayoutUserUID();
    if (sampler_count_pixel) {
      if (current_sampler_layout_uid_pixel_ != sampler_layout_uid_pixel) {
        current_sampler_layout_uid_pixel_ = sampler_layout_uid_pixel;
        cbuffer_binding_descriptor_indices_pixel_up_to_date_ = false;
      }
      current_samplers_pixel_.resize(
          std::max(current_samplers_pixel_.size(), sampler_count_pixel));
      for (size_t i = 0; i < sampler_count_pixel; ++i) {
        auto parameters =
            texture_cache_->GetSamplerParameters(sampler_bindings_pixel[i]);
        if (current_samplers_pixel_[i] != parameters) {
          current_samplers_pixel_[i] = parameters;
          cbuffer_binding_descriptor_indices_pixel_up_to_date_ = false;
        }
      }
    } else if (current_sampler_layout_uid_pixel_ != sampler_layout_uid_pixel) {
      current_sampler_layout_uid_pixel_ = sampler_layout_uid_pixel;
      cbuffer_binding_descriptor_indices_pixel_up_to_date_ = false;
    }
    if (current_texture_layout_uid_pixel_ != texture_layout_uid_pixel &&
        !texture_count_pixel) {
      cbuffer_binding_descriptor_indices_pixel_up_to_date_ = false;
    } else if (texture_count_pixel &&
               cbuffer_binding_descriptor_indices_pixel_up_to_date_ &&
               (current_texture_layout_uid_pixel_ != texture_layout_uid_pixel ||
                !texture_cache_->AreActiveTextureSRVKeysUpToDate(
                    current_texture_srv_keys_pixel_.data(),
                    texture_bindings_pixel.data(), texture_count_pixel))) {
      cbuffer_binding_descriptor_indices_pixel_up_to_date_ = false;
    }
    if (!cbuffer_binding_descriptor_indices_pixel_up_to_date_) {
      next_texture_bindless_indices_pixel.reserve(texture_count_pixel);
      for (const auto& binding : texture_bindings_pixel) {
        next_texture_bindless_indices_pixel.push_back(
            texture_cache_->GetBindlessSRVIndexForBinding(
                binding.fetch_constant, binding.dimension, binding.is_signed));
      }
      next_sampler_bindless_indices_pixel.reserve(sampler_count_pixel);
      for (const auto& binding : sampler_bindings_pixel) {
        next_sampler_bindless_indices_pixel.push_back(
            texture_cache_->GetBindlessSamplerIndexForBinding(binding));
      }
    }
  }

  // Determine whether any CBV needs re-uploading.
  bool any_cbuffer_dirty =
      !cbuffer_binding_system_up_to_date_ ||
      !cbuffer_binding_float_vertex_.up_to_date ||
      !cbuffer_binding_float_pixel_.up_to_date ||
      !cbuffer_binding_bool_loop_.up_to_date ||
      !cbuffer_binding_fetch_.up_to_date ||
      !cbuffer_binding_descriptor_indices_vertex_up_to_date_ ||
      !cbuffer_binding_descriptor_indices_pixel_up_to_date_;

  // Previous VS/PS table-start pointers (for copying unchanged CBV
  // regions).  The binding stores the TABLE-START offset, so contents() +
  // offset gives the b0 position of the previous allocation.
  uint8_t* prev_vs_data =
      cbuffer_binding_float_vertex_.buffer
          ? static_cast<uint8_t*>(
                cbuffer_binding_float_vertex_.buffer->contents()) +
                cbuffer_binding_float_vertex_.offset
          : nullptr;
  uint8_t* prev_ps_data =
      cbuffer_binding_float_pixel_.buffer
          ? static_cast<uint8_t*>(
                cbuffer_binding_float_pixel_.buffer->contents()) +
                cbuffer_binding_float_pixel_.offset
          : nullptr;

  // Vertex-stage uniforms buffer, offset, and GPU address for this draw.
  MTL::Buffer* vs_uniforms_buf = cbuffer_binding_float_vertex_.buffer;
  NS::UInteger vs_uniforms_off = cbuffer_binding_float_vertex_.offset;
  uint64_t vs_uniforms_gpu = cbuffer_binding_float_vertex_.gpu_address;
  // Pixel-stage uniforms buffer, offset, and GPU address for this draw.
  MTL::Buffer* ps_uniforms_buf = cbuffer_binding_float_pixel_.buffer;
  NS::UInteger ps_uniforms_off = cbuffer_binding_float_pixel_.offset;
  uint64_t ps_uniforms_gpu = cbuffer_binding_float_pixel_.gpu_address;
  bool descriptor_indices_vertex_written = false;
  bool descriptor_indices_pixel_written = false;

  if (any_cbuffer_dirty) {
    // Allocate one contiguous block per stage from the pool.
    constexpr size_t kConstantBufferAlignment = 256;
    uint64_t submission = submission_current_ ? submission_current_ : 1;

    MTL::Buffer* vs_buf = nullptr;
    size_t vs_off = 0;
    uint64_t vs_gpu = 0;
    uint8_t* vs_data = constant_buffer_pool_->Request(
        submission, kUniformsBytesPerTable, kConstantBufferAlignment, &vs_buf,
        vs_off, vs_gpu);

    MTL::Buffer* ps_buf = nullptr;
    size_t ps_off = 0;
    uint64_t ps_gpu = 0;
    uint8_t* ps_data = constant_buffer_pool_->Request(
        submission, kUniformsBytesPerTable, kConstantBufferAlignment, &ps_buf,
        ps_off, ps_gpu);

    if (!vs_data || !ps_data) {
      XELOGE("IssueDraw: constant buffer pool allocation failed");
      return false;
    }

    // b0: System constants.
    if (!cbuffer_binding_system_up_to_date_ || !prev_vs_data || !prev_ps_data) {
      std::memcpy(vs_data, &system_constants_,
                  sizeof(DxbcShaderTranslator::SystemConstants));
      std::memcpy(ps_data, &system_constants_,
                  sizeof(DxbcShaderTranslator::SystemConstants));
    } else {
      std::memcpy(vs_data, prev_vs_data,
                  sizeof(DxbcShaderTranslator::SystemConstants));
      std::memcpy(ps_data, prev_ps_data,
                  sizeof(DxbcShaderTranslator::SystemConstants));
    }

    // b1: Packed float constants.
    auto write_packed_float_constants = [&](uint8_t* dst, const Shader& shader,
                                            uint32_t regs_base) {
      std::memset(dst, 0, kCBVSize);
      const Shader::ConstantRegisterMap& map = shader.constant_register_map();
      if (!map.float_count) {
        return;
      }
      uint8_t* out = dst;
      for (uint32_t i = 0; i < 4; ++i) {
        uint64_t bits = map.float_bitmap[i];
        uint32_t constant_index;
        while (xe::bit_scan_forward(bits, &constant_index)) {
          bits &= ~(uint64_t(1) << constant_index);
          if (out + 4 * sizeof(uint32_t) > dst + kCBVSize) {
            return;
          }
          std::memcpy(
              out, &regs.values[regs_base + (i << 8) + (constant_index << 2)],
              4 * sizeof(uint32_t));
          out += 4 * sizeof(uint32_t);
        }
      }
    };

    if (!cbuffer_binding_float_vertex_.up_to_date) {
      write_packed_float_constants(vs_data + kFloatConstantOffset,
                                   *vertex_shader,
                                   XE_GPU_REG_SHADER_CONSTANT_000_X);
    } else if (prev_vs_data) {
      std::memcpy(vs_data + kFloatConstantOffset,
                  prev_vs_data + kFloatConstantOffset, kCBVSize);
    } else {
      std::memset(vs_data + kFloatConstantOffset, 0, kCBVSize);
    }

    if (pixel_shader) {
      if (!cbuffer_binding_float_pixel_.up_to_date) {
        write_packed_float_constants(ps_data + kFloatConstantOffset,
                                     *pixel_shader,
                                     XE_GPU_REG_SHADER_CONSTANT_256_X);
      } else if (prev_ps_data) {
        std::memcpy(ps_data + kFloatConstantOffset,
                    prev_ps_data + kFloatConstantOffset, kCBVSize);
      } else {
        std::memset(ps_data + kFloatConstantOffset, 0, kCBVSize);
      }
    } else {
      std::memset(ps_data + kFloatConstantOffset, 0, kCBVSize);
    }

    // b2: Bool/loop constants.
    if (!cbuffer_binding_bool_loop_.up_to_date) {
      std::memcpy(vs_data + kBoolLoopConstantOffset,
                  &regs.values[XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031],
                  kBoolLoopConstantsSize);
      std::memcpy(ps_data + kBoolLoopConstantOffset,
                  &regs.values[XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031],
                  kBoolLoopConstantsSize);
    } else {
      if (prev_vs_data) {
        std::memcpy(vs_data + kBoolLoopConstantOffset,
                    prev_vs_data + kBoolLoopConstantOffset,
                    kBoolLoopConstantsSize);
      }
      if (prev_ps_data) {
        std::memcpy(ps_data + kBoolLoopConstantOffset,
                    prev_ps_data + kBoolLoopConstantOffset,
                    kBoolLoopConstantsSize);
      }
    }

    // b3: Fetch constants.
    if (!cbuffer_binding_fetch_.up_to_date) {
      std::memcpy(vs_data + kFetchConstantOffset,
                  &regs.values[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0],
                  kFetchConstantCount * sizeof(uint32_t));
      std::memcpy(ps_data + kFetchConstantOffset,
                  &regs.values[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0],
                  kFetchConstantCount * sizeof(uint32_t));
    } else {
      if (prev_vs_data) {
        std::memcpy(vs_data + kFetchConstantOffset,
                    prev_vs_data + kFetchConstantOffset,
                    kFetchConstantCount * sizeof(uint32_t));
      }
      if (prev_ps_data) {
        std::memcpy(ps_data + kFetchConstantOffset,
                    prev_ps_data + kFetchConstantOffset,
                    kFetchConstantCount * sizeof(uint32_t));
      }
    }

    // b4: Descriptor indices for bindless access.
    // Each shader texture/sampler binding reads an index from b4 and uses
    // it to look up a persistent bindless heap slot.
    //   - Textures: b4[texture.bindless_descriptor_index] =
    //       persistent SRV index from the view_bindless_heap_.
    //   - Samplers: b4[sampler.bindless_descriptor_index] =
    //       persistent sampler index from the sampler_bindless_heap_.
    {
      auto fill_descriptor_indices =
          [&](MetalShader* shader, uint8_t* data,
              const std::vector<uint32_t>& texture_indices,
              const std::vector<uint32_t>& sampler_indices) {
            std::memset(data + 4 * kCBVSize, 0,
                        kUniformsBytesPerTable - 4 * kCBVSize);
            if (!shader || !texture_cache_) {
              return;
            }
            auto* indices = reinterpret_cast<uint32_t*>(data + 4 * kCBVSize);
            const auto& tex_bindings =
                shader->GetTextureBindingsAfterTranslation();
            for (size_t i = 0;
                 i < tex_bindings.size() && i < texture_indices.size(); ++i) {
              uint32_t d = tex_bindings[i].bindless_descriptor_index;
              if (d >= kCBVSize / sizeof(uint32_t)) continue;
              indices[d] = texture_indices[i];
            }
            const auto& smp_bindings =
                shader->GetSamplerBindingsAfterTranslation();
            for (size_t i = 0;
                 i < smp_bindings.size() && i < sampler_indices.size(); ++i) {
              uint32_t d = smp_bindings[i].bindless_descriptor_index;
              if (d >= kCBVSize / sizeof(uint32_t)) continue;
              indices[d] = sampler_indices[i];
            }
          };
      if (!cbuffer_binding_descriptor_indices_vertex_up_to_date_ ||
          !prev_vs_data) {
        fill_descriptor_indices(metal_vertex_shader, vs_data,
                                next_texture_bindless_indices_vertex,
                                next_sampler_bindless_indices_vertex);
        descriptor_indices_vertex_written = true;
      } else {
        std::memcpy(vs_data + 4 * kCBVSize, prev_vs_data + 4 * kCBVSize,
                    kUniformsBytesPerTable - 4 * kCBVSize);
      }
      if (!cbuffer_binding_descriptor_indices_pixel_up_to_date_ ||
          !prev_ps_data) {
        fill_descriptor_indices(metal_pixel_shader, ps_data,
                                next_texture_bindless_indices_pixel,
                                next_sampler_bindless_indices_pixel);
        descriptor_indices_pixel_written = true;
      } else {
        std::memcpy(ps_data + 4 * kCBVSize, prev_ps_data + 4 * kCBVSize,
                    kUniformsBytesPerTable - 4 * kCBVSize);
      }
    }

    // Update cached bindings.  The offset and gpu_address stored in the
    // float_vertex/float_pixel bindings are the TABLE START (b0 position)
    // so that prev_vs_data/prev_ps_data can reconstruct the full table
    // pointer for copying unchanged CBV regions.
    cbuffer_binding_float_vertex_ = {vs_buf, vs_off, vs_gpu, true};
    cbuffer_binding_float_pixel_ = {ps_buf, ps_off, ps_gpu, true};
    cbuffer_binding_system_up_to_date_ = true;
    cbuffer_binding_bool_loop_.up_to_date = true;
    cbuffer_binding_fetch_.up_to_date = true;
    cbuffer_binding_descriptor_indices_vertex_up_to_date_ = true;
    cbuffer_binding_descriptor_indices_pixel_up_to_date_ = true;
    if (descriptor_indices_vertex_written) {
      current_texture_layout_uid_vertex_ = texture_layout_uid_vertex;
      current_texture_bindless_indices_vertex_ =
          std::move(next_texture_bindless_indices_vertex);
      current_sampler_bindless_indices_vertex_ =
          std::move(next_sampler_bindless_indices_vertex);
      if (texture_count_vertex) {
        current_texture_srv_keys_vertex_.resize(std::max(
            current_texture_srv_keys_vertex_.size(), texture_count_vertex));
        texture_cache_->WriteActiveTextureSRVKeys(
            current_texture_srv_keys_vertex_.data(),
            texture_bindings_vertex.data(), texture_count_vertex);
      }
    }
    if (descriptor_indices_pixel_written) {
      current_texture_layout_uid_pixel_ = texture_layout_uid_pixel;
      current_texture_bindless_indices_pixel_ =
          std::move(next_texture_bindless_indices_pixel);
      current_sampler_bindless_indices_pixel_ =
          std::move(next_sampler_bindless_indices_pixel);
      if (texture_bindings_pixel_ptr && !texture_bindings_pixel_ptr->empty()) {
        current_texture_srv_keys_pixel_.resize(
            std::max(current_texture_srv_keys_pixel_.size(),
                     texture_bindings_pixel_ptr->size()));
        texture_cache_->WriteActiveTextureSRVKeys(
            current_texture_srv_keys_pixel_.data(),
            texture_bindings_pixel_ptr->data(),
            texture_bindings_pixel_ptr->size());
      }
    }

    vs_uniforms_buf = vs_buf;
    vs_uniforms_off = static_cast<NS::UInteger>(vs_off);
    vs_uniforms_gpu = vs_gpu;
    ps_uniforms_buf = ps_buf;
    ps_uniforms_off = static_cast<NS::UInteger>(ps_off);
    ps_uniforms_gpu = ps_gpu;
  }

  uniforms_out.vs_buf = vs_uniforms_buf;
  uniforms_out.vs_off = vs_uniforms_off;
  uniforms_out.vs_gpu = vs_uniforms_gpu;
  uniforms_out.ps_buf = ps_uniforms_buf;
  uniforms_out.ps_off = ps_uniforms_off;
  uniforms_out.ps_gpu = ps_uniforms_gpu;
  return true;
}

bool MetalCommandProcessor::PopulateBindlessTables(
    MetalShader* metal_vertex_shader, MetalShader* metal_pixel_shader,
    bool shared_memory_is_uav, MTL::ResourceUsage shared_memory_usage,
    bool use_geometry_emulation, bool use_tessellation_emulation,
    const UniformBufferInfo& uniforms) {
  constexpr size_t kStageVertex = 0;
  constexpr size_t kStagePixel = 1;
  constexpr size_t kBindlessTableCount = kStageCount;
  constexpr size_t kBindlessCBVTableBytes = kBindlessTableCount *
                                            kCbvHeapSlotsPerTable *
                                            sizeof(IRDescriptorTableEntry);
  constexpr size_t kBindlessTopLevelTableBytes =
      kBindlessTableCount * kTopLevelABBytesPerTable;

  uint64_t uniforms_gpu_base_vertex = uniforms.vs_gpu;
  uint64_t uniforms_gpu_base_pixel = uniforms.ps_gpu;
  bool reuse_bindless_table =
      current_bindless_table_valid_ &&
      current_bindless_vs_uniforms_gpu_ == uniforms_gpu_base_vertex &&
      current_bindless_ps_uniforms_gpu_ == uniforms_gpu_base_pixel &&
      current_bindless_shared_memory_is_uav_ == shared_memory_is_uav &&
      current_bindless_uses_mesh_stages_ ==
          (use_geometry_emulation || use_tessellation_emulation);

  if (!reuse_bindless_table) {
    uint64_t submission = submission_current_ ? submission_current_ : 1;
    MTL::Buffer* cbv_table_buffer = nullptr;
    size_t cbv_table_offset = 0;
    uint64_t cbv_table_gpu_address = 0;
    auto* cbv_entries_all = reinterpret_cast<IRDescriptorTableEntry*>(
        constant_buffer_pool_->Request(
            submission, kBindlessCBVTableBytes, kCbvSizeBytes,
            &cbv_table_buffer, cbv_table_offset, cbv_table_gpu_address));
    MTL::Buffer* top_level_buffer = nullptr;
    size_t top_level_offset = 0;
    uint64_t top_level_gpu_address = 0;
    auto* top_level_entries_all =
        reinterpret_cast<uint64_t*>(constant_buffer_pool_->Request(
            submission, kBindlessTopLevelTableBytes, kTopLevelABBytesPerTable,
            &top_level_buffer, top_level_offset, top_level_gpu_address));
    if (!cbv_entries_all || !top_level_entries_all) {
      XELOGE("IssueDraw: bindless table allocation failed");
      return false;
    }

    constexpr uint64_t kDescriptorEntrySize = sizeof(IRDescriptorTableEntry);
    uint64_t view_heap_gpu = view_bindless_heap_->gpuAddress();
    uint64_t sampler_heap_gpu = sampler_bindless_heap_->gpuAddress();
    uint64_t system_view_gpu = system_view_tables_->gpuAddress();
    uint64_t srv_space0_gpu =
        system_view_gpu + (shared_memory_is_uav
                               ? kSystemViewTableSRVNull
                               : kSystemViewTableSRVSharedMemory) *
                              kDescriptorEntrySize;
    uint64_t uav_space0_gpu =
        system_view_gpu + (shared_memory_is_uav
                               ? kSystemViewTableUAVSharedMemoryStart
                               : kSystemViewTableUAVNullStart) *
                              kDescriptorEntrySize;
    uint64_t null_uav_gpu =
        system_view_gpu + kSystemViewTableUAVNullStart * kDescriptorEntrySize;

    auto write_top_level_and_cbvs_bindless =
        [&](size_t stage_index, IRDescriptorTableEntry* cbv_entries,
            uint64_t uniforms_gpu_base) {
          auto* top_level_ptrs = reinterpret_cast<uint64_t*>(
              reinterpret_cast<uint8_t*>(top_level_entries_all) +
              stage_index * kTopLevelABBytesPerTable);
          std::memset(top_level_ptrs, 0, kTopLevelABBytesPerTable);

          top_level_ptrs[0] = srv_space0_gpu;
          top_level_ptrs[5] = uav_space0_gpu;
          top_level_ptrs[1] = view_heap_gpu;
          top_level_ptrs[2] = view_heap_gpu;
          top_level_ptrs[3] = view_heap_gpu;
          top_level_ptrs[4] = view_heap_gpu;
          top_level_ptrs[6] = null_uav_gpu;
          top_level_ptrs[7] = null_uav_gpu;
          top_level_ptrs[8] = null_uav_gpu;
          top_level_ptrs[9] = sampler_heap_gpu;

          IRDescriptorTableSetBuffer(&cbv_entries[0],
                                     uniforms_gpu_base + 0 * kCbvSizeBytes,
                                     kCbvSizeBytes);
          IRDescriptorTableSetBuffer(&cbv_entries[1],
                                     uniforms_gpu_base + 1 * kCbvSizeBytes,
                                     kCbvSizeBytes);
          IRDescriptorTableSetBuffer(&cbv_entries[2],
                                     uniforms_gpu_base + 2 * kCbvSizeBytes,
                                     kCbvSizeBytes);
          IRDescriptorTableSetBuffer(&cbv_entries[3],
                                     uniforms_gpu_base + 3 * kCbvSizeBytes,
                                     kCbvSizeBytes);
          IRDescriptorTableSetBuffer(&cbv_entries[4],
                                     uniforms_gpu_base + 4 * kCbvSizeBytes,
                                     kCbvSizeBytes);
          IRDescriptorTableSetBuffer(&cbv_entries[5],
                                     null_buffer_->gpuAddress(), kCbvSizeBytes);
          IRDescriptorTableSetBuffer(&cbv_entries[6],
                                     null_buffer_->gpuAddress(), kCbvSizeBytes);

          uint64_t cbv_table_gpu_base =
              cbv_table_gpu_address +
              stage_index * kCbvHeapSlotsPerTable * kDescriptorEntrySize;
          top_level_ptrs[10] = cbv_table_gpu_base;
          top_level_ptrs[11] = cbv_table_gpu_base;
          top_level_ptrs[12] = cbv_table_gpu_base;
          top_level_ptrs[13] = cbv_table_gpu_base;
        };

    write_top_level_and_cbvs_bindless(
        kStageVertex, cbv_entries_all + kStageVertex * kCbvHeapSlotsPerTable,
        uniforms_gpu_base_vertex);
    write_top_level_and_cbvs_bindless(
        kStagePixel, cbv_entries_all + kStagePixel * kCbvHeapSlotsPerTable,
        uniforms_gpu_base_pixel);

    current_bindless_table_valid_ = true;
    current_bindless_top_level_buffer_ = top_level_buffer;
    current_bindless_top_level_offset_ =
        static_cast<NS::UInteger>(top_level_offset);
    current_bindless_top_level_gpu_address_ = top_level_gpu_address;
    current_bindless_cbv_buffer_ = cbv_table_buffer;
    current_bindless_cbv_offset_ = static_cast<NS::UInteger>(cbv_table_offset);
    current_bindless_cbv_gpu_address_ = cbv_table_gpu_address;
    current_bindless_vs_uniforms_gpu_ = uniforms_gpu_base_vertex;
    current_bindless_ps_uniforms_gpu_ = uniforms_gpu_base_pixel;
    current_bindless_shared_memory_is_uav_ = shared_memory_is_uav;
    current_bindless_uses_mesh_stages_ =
        use_geometry_emulation || use_tessellation_emulation;

    MTL::Buffer* shared_mem_buffer = shared_memory_->GetBuffer();
    if (shared_mem_buffer) {
      UseRenderEncoderResource(shared_mem_buffer, shared_memory_usage);
    }
    if (render_target_cache_) {
      render_target_cache_->UseBindlessResources(
          *this, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
    }

    std::array<MTL::Texture*, 64> textures_for_encoder;
    uint32_t textures_for_encoder_count = 0;
    auto track_texture_usage = [&](MTL::Texture* texture) {
      if (!texture) {
        return;
      }
      for (uint32_t i = 0; i < textures_for_encoder_count; ++i) {
        if (textures_for_encoder[i] == texture) {
          return;
        }
      }
      assert_true(textures_for_encoder_count < textures_for_encoder.size());
      textures_for_encoder[textures_for_encoder_count++] = texture;
    };

    auto track_shader_texture_usage = [&](MetalShader* shader) {
      if (!shader || !texture_cache_) {
        return;
      }
      const auto& shader_texture_bindings =
          shader->GetTextureBindingsAfterTranslation();
      MetalTextureCache* metal_texture_cache = texture_cache_.get();
      for (const auto& binding : shader_texture_bindings) {
        MTL::Texture* texture = texture_cache_->GetTextureForBinding(
            binding.fetch_constant, binding.dimension, binding.is_signed);
        if (!texture) {
          switch (binding.dimension) {
            case xenos::FetchOpDimension::k1D:
            case xenos::FetchOpDimension::k2D:
              texture = metal_texture_cache->GetNullTexture2D();
              break;
            case xenos::FetchOpDimension::k3DOrStacked:
              texture = metal_texture_cache->GetNullTexture3D();
              break;
            case xenos::FetchOpDimension::kCube:
              texture = metal_texture_cache->GetNullTextureCube();
              break;
            default:
              texture = metal_texture_cache->GetNullTexture2D();
              break;
          }
        }
        if (texture) {
          track_texture_usage(texture);
        }
      }
    };

    track_shader_texture_usage(metal_vertex_shader);
    track_shader_texture_usage(metal_pixel_shader);

    for (uint32_t i = 0; i < textures_for_encoder_count; ++i) {
      UseRenderEncoderResource(textures_for_encoder[i], MTL::ResourceUsageRead);
    }

    UseRenderEncoderResource(null_buffer_, MTL::ResourceUsageRead);
    UseRenderEncoderResource(view_bindless_heap_, MTL::ResourceUsageRead);
    UseRenderEncoderResource(sampler_bindless_heap_, MTL::ResourceUsageRead);
    UseRenderEncoderResource(system_view_tables_, MTL::ResourceUsageRead);
    UseRenderEncoderResource(current_bindless_top_level_buffer_,
                             MTL::ResourceUsageRead);
    UseRenderEncoderResource(current_bindless_cbv_buffer_,
                             MTL::ResourceUsageRead);
    if (uniforms.vs_buf) {
      UseRenderEncoderResource(uniforms.vs_buf, MTL::ResourceUsageRead);
    }
    if (uniforms.ps_buf && uniforms.ps_buf != uniforms.vs_buf) {
      UseRenderEncoderResource(uniforms.ps_buf, MTL::ResourceUsageRead);
    }

    const NS::UInteger top_level_offset_vertex =
        current_bindless_top_level_offset_ +
        NS::UInteger(kStageVertex * kTopLevelABBytesPerTable);
    const NS::UInteger top_level_offset_pixel =
        current_bindless_top_level_offset_ +
        NS::UInteger(kStagePixel * kTopLevelABBytesPerTable);
    if (use_geometry_emulation || use_tessellation_emulation) {
      current_render_encoder_->setObjectBuffer(
          current_bindless_top_level_buffer_, top_level_offset_vertex,
          kIRArgumentBufferBindPoint);
      current_render_encoder_->setMeshBuffer(current_bindless_top_level_buffer_,
                                             top_level_offset_vertex,
                                             kIRArgumentBufferBindPoint);
      current_render_encoder_->setFragmentBuffer(
          current_bindless_top_level_buffer_, top_level_offset_pixel,
          kIRArgumentBufferBindPoint);

      if (use_tessellation_emulation) {
        current_render_encoder_->setObjectBuffer(
            current_bindless_top_level_buffer_, top_level_offset_vertex,
            kIRArgumentBufferHullDomainBindPoint);
        current_render_encoder_->setMeshBuffer(
            current_bindless_top_level_buffer_, top_level_offset_vertex,
            kIRArgumentBufferHullDomainBindPoint);
      }

      if (uniforms.vs_buf) {
        current_render_encoder_->setObjectBuffer(
            uniforms.vs_buf, uniforms.vs_off,
            kIRArgumentBufferUniformsBindPoint);
        current_render_encoder_->setMeshBuffer(
            uniforms.vs_buf, uniforms.vs_off,
            kIRArgumentBufferUniformsBindPoint);
      }
      if (uniforms.ps_buf) {
        current_render_encoder_->setFragmentBuffer(
            uniforms.ps_buf, uniforms.ps_off,
            kIRArgumentBufferUniformsBindPoint);
      }

      if (!heap_binds_set_on_encoder_) {
        current_render_encoder_->setObjectBuffer(view_bindless_heap_, 0,
                                                 kIRDescriptorHeapBindPoint);
        current_render_encoder_->setMeshBuffer(view_bindless_heap_, 0,
                                               kIRDescriptorHeapBindPoint);
        current_render_encoder_->setFragmentBuffer(view_bindless_heap_, 0,
                                                   kIRDescriptorHeapBindPoint);
        current_render_encoder_->setObjectBuffer(sampler_bindless_heap_, 0,
                                                 kIRSamplerHeapBindPoint);
        current_render_encoder_->setMeshBuffer(sampler_bindless_heap_, 0,
                                               kIRSamplerHeapBindPoint);
        current_render_encoder_->setFragmentBuffer(sampler_bindless_heap_, 0,
                                                   kIRSamplerHeapBindPoint);
        heap_binds_set_on_encoder_ = true;
      }
    } else {
      current_render_encoder_->setVertexBuffer(
          current_bindless_top_level_buffer_, top_level_offset_vertex,
          kIRArgumentBufferBindPoint);
      current_render_encoder_->setFragmentBuffer(
          current_bindless_top_level_buffer_, top_level_offset_pixel,
          kIRArgumentBufferBindPoint);

      if (uniforms.vs_buf) {
        current_render_encoder_->setVertexBuffer(
            uniforms.vs_buf, uniforms.vs_off,
            kIRArgumentBufferUniformsBindPoint);
      }
      if (uniforms.ps_buf) {
        current_render_encoder_->setFragmentBuffer(
            uniforms.ps_buf, uniforms.ps_off,
            kIRArgumentBufferUniformsBindPoint);
      }

      if (!heap_binds_set_on_encoder_) {
        current_render_encoder_->setVertexBuffer(view_bindless_heap_, 0,
                                                 kIRDescriptorHeapBindPoint);
        current_render_encoder_->setFragmentBuffer(view_bindless_heap_, 0,
                                                   kIRDescriptorHeapBindPoint);
        current_render_encoder_->setVertexBuffer(sampler_bindless_heap_, 0,
                                                 kIRSamplerHeapBindPoint);
        current_render_encoder_->setFragmentBuffer(sampler_bindless_heap_, 0,
                                                   kIRSamplerHeapBindPoint);
        heap_binds_set_on_encoder_ = true;
      }
    }
  }

  return true;
}

bool MetalCommandProcessor::DispatchDraw(
    const RegisterFile& regs,
    const PrimitiveProcessor::ProcessingResult& primitive_processing_result,
    bool use_tessellation_emulation,
    MetalPipelineCache::TessellationPipelineState* tessellation_pipeline_state,
    bool use_geometry_emulation,
    MetalPipelineCache::GeometryPipelineState* geometry_pipeline_state,
    bool shared_memory_is_uav, MTL::ResourceUsage shared_memory_usage,
    bool memexport_used, bool uses_vertex_fetch,
    const std::vector<Shader::VertexBinding>& vb_bindings,
    const VertexBindingRange* vertex_ranges, uint32_t vertex_range_count,
    IndexBufferInfo* index_buffer_info) {
  // Bind vertex buffers / descriptors.
  if (use_geometry_emulation || use_tessellation_emulation) {
    IRRuntimeVertexBuffers vertex_buffers = {};
    MTL::Buffer* shared_mem_buffer =
        shared_memory_ ? shared_memory_->GetBuffer() : nullptr;
    if (shared_mem_buffer) {
      UseRenderEncoderResource(shared_mem_buffer, shared_memory_usage);
      for (uint32_t i = 0; i < vertex_range_count; ++i) {
        const auto& range = vertex_ranges[i];
        size_t binding_index = range.binding_index;
        if (binding_index <
            (sizeof(vertex_buffers) / sizeof(vertex_buffers[0]))) {
          vertex_buffers[binding_index].addr =
              shared_mem_buffer->gpuAddress() + range.offset;
          vertex_buffers[binding_index].length = range.length;
          vertex_buffers[binding_index].stride = range.stride;
        }
      }
    }
    // MSC manual: bind IRRuntimeVertexBuffers at kIRVertexBufferBindPoint (6)
    // for the object stage when using geometry emulation.
    current_render_encoder_->setObjectBytes(
        vertex_buffers, sizeof(vertex_buffers), kIRVertexBufferBindPoint);
  } else if (uses_vertex_fetch) {
    // Vertex fetch shaders read directly from shared memory via SRV, so avoid
    // stage-in bindings that can trigger invalid buffer loads.
    if (shared_memory_) {
      if (MTL::Buffer* shared_mem_buffer = shared_memory_->GetBuffer()) {
        UseRenderEncoderResource(shared_mem_buffer, shared_memory_usage);
      }
    }
  } else {
    // Bind vertex buffers at kIRVertexBufferBindPoint (index 6+) for stage-in.
    // The pipeline's vertex descriptor expects buffers at these indices,
    // populated from the vertex fetch constants. The buffer addresses come from
    // shared memory.
    if (shared_memory_ && !vb_bindings.empty()) {
      MTL::Buffer* shared_mem_buffer = shared_memory_->GetBuffer();
      if (shared_mem_buffer) {
        // Mark shared memory as used for reading
        UseRenderEncoderResource(shared_mem_buffer, shared_memory_usage);

        // Bind vertex buffers for each binding
        for (uint32_t i = 0; i < vertex_range_count; ++i) {
          const auto& range = vertex_ranges[i];
          uint64_t buffer_index =
              kIRVertexBufferBindPoint + uint64_t(range.binding_index);
          current_render_encoder_->setVertexBuffer(shared_mem_buffer,
                                                   range.offset, buffer_index);
        }
      }
    } else if (shared_memory_) {
      // No vertex bindings, but still mark shared memory as resident
      if (MTL::Buffer* shared_mem_buffer = shared_memory_->GetBuffer()) {
        UseRenderEncoderResource(shared_mem_buffer, shared_memory_usage);
      }
    }
  }

  auto request_guest_index_range = [&](uint64_t index_base,
                                       uint32_t index_count,
                                       MTL::IndexType index_type) -> bool {
    if (!shared_memory_) {
      return false;
    }
    uint32_t index_stride = (index_type == MTL::IndexTypeUInt16)
                                ? sizeof(uint16_t)
                                : sizeof(uint32_t);
    uint64_t index_length = uint64_t(index_count) * index_stride;
    if (index_base > SharedMemory::kBufferSize ||
        SharedMemory::kBufferSize - index_base < index_length) {
      XELOGW(
          "Index buffer range out of bounds (base=0x{:08X} size={} count={})",
          static_cast<uint32_t>(index_base), index_length, index_count);
      return false;
    }
    return shared_memory_->RequestRange(static_cast<uint32_t>(index_base),
                                        static_cast<uint32_t>(index_length));
  };
  auto resolve_guest_dma_index_buffer =
      [&](uint64_t guest_index_base, uint32_t index_count,
          MTL::IndexType index_type, MTL::Buffer*& index_buffer_out,
          uint64_t& index_offset_out) -> bool {
    index_buffer_out = nullptr;
    index_offset_out = 0;
    if (!request_guest_index_range(guest_index_base, index_count, index_type)) {
      return false;
    }
    MTL::Buffer* shared_mem_buffer =
        shared_memory_ ? shared_memory_->GetBuffer() : nullptr;
    if (!shared_mem_buffer) {
      return false;
    }
    if (!memexport_used) {
      index_buffer_out = shared_mem_buffer;
      index_offset_out = guest_index_base;
      return true;
    }
    uint32_t index_stride = (index_type == MTL::IndexTypeUInt16)
                                ? sizeof(uint16_t)
                                : sizeof(uint32_t);
    size_t index_bytes = size_t(index_count) * index_stride;
    uint64_t submission = submission_current_ ? submission_current_ : 1;
    MTL::Buffer* scratch_buffer = nullptr;
    size_t scratch_offset = 0;
    uint64_t scratch_gpu_address = 0;
    uint8_t* scratch_mapping = constant_buffer_pool_->Request(
        submission, index_bytes, index_stride, &scratch_buffer, scratch_offset,
        scratch_gpu_address);
    if (!scratch_mapping || !scratch_buffer) {
      XELOGE(
          "IssueDraw: failed to allocate scratch index buffer for guest DMA "
          "memexport draw");
      return false;
    }
    const uint8_t* shared_memory_bytes =
        static_cast<const uint8_t*>(shared_mem_buffer->contents());
    std::memcpy(scratch_mapping, shared_memory_bytes + guest_index_base,
                index_bytes);
    index_buffer_out = scratch_buffer;
    index_offset_out = scratch_offset;
    return true;
  };

  // Shared index buffer resolution used by tessellation, geometry, and
  // standard indexed draw paths.  Returns false on fatal error.
  auto resolve_index_buffer = [&](MTL::IndexType index_type,
                                  MTL::Buffer*& index_buffer_out,
                                  uint64_t& index_offset_out) -> bool {
    index_buffer_out = nullptr;
    index_offset_out = 0;
    switch (primitive_processing_result.index_buffer_type) {
      case PrimitiveProcessor::ProcessedIndexBufferType::kGuestDMA:
        if (!resolve_guest_dma_index_buffer(
                primitive_processing_result.guest_index_base,
                primitive_processing_result.host_draw_vertex_count, index_type,
                index_buffer_out, index_offset_out)) {
          XELOGE("IssueDraw: failed to resolve guest DMA index buffer");
          return false;
        }
        break;
      case PrimitiveProcessor::ProcessedIndexBufferType::kHostConverted:
        if (primitive_processor_) {
          index_buffer_out = primitive_processor_->GetConvertedIndexBuffer(
              primitive_processing_result.host_index_buffer_handle,
              index_offset_out);
        }
        break;
      case PrimitiveProcessor::ProcessedIndexBufferType::kHostBuiltinForAuto:
      case PrimitiveProcessor::ProcessedIndexBufferType::kHostBuiltinForDMA:
        if (primitive_processor_) {
          index_buffer_out = primitive_processor_->GetBuiltinIndexBuffer();
          index_offset_out =
              primitive_processing_result.host_index_buffer_handle;
        }
        break;
      default:
        XELOGE("Unsupported index buffer type {}",
               uint32_t(primitive_processing_result.index_buffer_type));
        return false;
    }
    if (!index_buffer_out) {
      XELOGE("IssueDraw: index buffer is null for type {}",
             uint32_t(primitive_processing_result.index_buffer_type));
      return false;
    }
    UseRenderEncoderResource(index_buffer_out, MTL::ResourceUsageRead);
    return true;
  };

  if (use_tessellation_emulation) {
    IRRuntimePrimitiveType tess_primitive = IRRuntimePrimitiveTypeTriangle;
    switch (primitive_processing_result.host_primitive_type) {
      case xenos::PrimitiveType::kTriangleList:
        tess_primitive = IRRuntimePrimitiveType3ControlPointPatchlist;
        break;
      case xenos::PrimitiveType::kQuadList:
        tess_primitive = IRRuntimePrimitiveType4ControlPointPatchlist;
        break;
      case xenos::PrimitiveType::kTrianglePatch:
        tess_primitive = (primitive_processing_result.tessellation_mode ==
                          xenos::TessellationMode::kAdaptive)
                             ? IRRuntimePrimitiveType3ControlPointPatchlist
                             : IRRuntimePrimitiveType1ControlPointPatchlist;
        break;
      case xenos::PrimitiveType::kQuadPatch:
        tess_primitive = (primitive_processing_result.tessellation_mode ==
                          xenos::TessellationMode::kAdaptive)
                             ? IRRuntimePrimitiveType4ControlPointPatchlist
                             : IRRuntimePrimitiveType1ControlPointPatchlist;
        break;
      default:
        XELOGE(
            "Host tessellated primitive type {} returned by the primitive "
            "processor is not supported by the Metal tessellation path",
            uint32_t(primitive_processing_result.host_primitive_type));
        return false;
    }

    const IRRuntimeTessellationPipelineConfig& tess_config =
        tessellation_pipeline_state->config;

    if (primitive_processing_result.index_buffer_type ==
        PrimitiveProcessor::ProcessedIndexBufferType::kNone) {
      IRRuntimeDrawPatchesTessellationEmulation(
          current_render_encoder_, tess_primitive, tess_config, 1,
          primitive_processing_result.host_draw_vertex_count, 0, 0);
    } else {
      MTL::IndexType index_type =
          (primitive_processing_result.host_index_format ==
           xenos::IndexFormat::kInt16)
              ? MTL::IndexTypeUInt16
              : MTL::IndexTypeUInt32;
      MTL::Buffer* index_buffer = nullptr;
      uint64_t index_offset = 0;
      if (!resolve_index_buffer(index_type, index_buffer, index_offset)) {
        return false;
      }
      uint32_t index_stride = (index_type == MTL::IndexTypeUInt16)
                                  ? sizeof(uint16_t)
                                  : sizeof(uint32_t);
      uint32_t start_index =
          index_stride ? uint32_t(index_offset / index_stride) : 0;
      IRRuntimeDrawIndexedPatchesTessellationEmulation(
          current_render_encoder_, tess_primitive, index_type, index_buffer,
          tess_config, 1, primitive_processing_result.host_draw_vertex_count, 0,
          0, start_index);
    }
  } else if (use_geometry_emulation) {
    IRRuntimePrimitiveType geometry_primitive = IRRuntimePrimitiveTypeTriangle;
    switch (primitive_processing_result.host_primitive_type) {
      case xenos::PrimitiveType::kPointList:
        geometry_primitive = IRRuntimePrimitiveTypePoint;
        break;
      case xenos::PrimitiveType::kRectangleList:
        geometry_primitive = IRRuntimePrimitiveTypeTriangle;
        break;
      case xenos::PrimitiveType::kQuadList:
        geometry_primitive = IRRuntimePrimitiveTypeLineWithAdj;
        break;
      default:
        XELOGE(
            "Host primitive type {} returned by the primitive processor is not "
            "supported by the Metal geometry path",
            uint32_t(primitive_processing_result.host_primitive_type));
        return false;
    }

    IRRuntimeGeometryPipelineConfig geometry_config = {};
    geometry_config.gsVertexSizeInBytes =
        geometry_pipeline_state->gs_vertex_size_in_bytes;
    geometry_config.gsMaxInputPrimitivesPerMeshThreadgroup =
        geometry_pipeline_state->gs_max_input_primitives_per_mesh_threadgroup;

    if (primitive_processing_result.index_buffer_type ==
        PrimitiveProcessor::ProcessedIndexBufferType::kNone) {
      IRRuntimeDrawPrimitivesGeometryEmulation(
          current_render_encoder_, geometry_primitive, geometry_config, 1,
          primitive_processing_result.host_draw_vertex_count, 0, 0);
    } else {
      MTL::IndexType index_type =
          (primitive_processing_result.host_index_format ==
           xenos::IndexFormat::kInt16)
              ? MTL::IndexTypeUInt16
              : MTL::IndexTypeUInt32;
      MTL::Buffer* index_buffer = nullptr;
      uint64_t index_offset = 0;
      if (!resolve_index_buffer(index_type, index_buffer, index_offset)) {
        return false;
      }
      uint32_t index_stride = (index_type == MTL::IndexTypeUInt16)
                                  ? sizeof(uint16_t)
                                  : sizeof(uint32_t);
      uint32_t start_index =
          index_stride ? uint32_t(index_offset / index_stride) : 0;
      IRRuntimeDrawIndexedPrimitivesGeometryEmulation(
          current_render_encoder_, geometry_primitive, index_type, index_buffer,
          geometry_config, 1,
          primitive_processing_result.host_draw_vertex_count, start_index, 0,
          0);
    }
  } else {
    // Primitive topology - from primitive processor, like D3D12.
    MTL::PrimitiveType mtl_primitive = MTL::PrimitiveTypeTriangle;
    switch (primitive_processing_result.host_primitive_type) {
      case xenos::PrimitiveType::kPointList:
        mtl_primitive = MTL::PrimitiveTypePoint;
        break;
      case xenos::PrimitiveType::kLineList:
        mtl_primitive = MTL::PrimitiveTypeLine;
        break;
      case xenos::PrimitiveType::kLineStrip:
        mtl_primitive = MTL::PrimitiveTypeLineStrip;
        break;
      case xenos::PrimitiveType::kTriangleList:
      case xenos::PrimitiveType::kRectangleList:
        mtl_primitive = MTL::PrimitiveTypeTriangle;
        break;
      case xenos::PrimitiveType::kTriangleStrip:
        mtl_primitive = MTL::PrimitiveTypeTriangleStrip;
        break;
      default:
        XELOGE(
            "Host primitive type {} returned by the primitive processor is not "
            "supported by the Metal command processor",
            uint32_t(primitive_processing_result.host_primitive_type));
        return false;
    }

    // Draw using primitive processor output.
    if (primitive_processing_result.index_buffer_type ==
        PrimitiveProcessor::ProcessedIndexBufferType::kNone) {
      IRRuntimeDrawPrimitives(
          current_render_encoder_, mtl_primitive, NS::UInteger(0),
          NS::UInteger(primitive_processing_result.host_draw_vertex_count));
    } else {
      MTL::IndexType index_type =
          (primitive_processing_result.host_index_format ==
           xenos::IndexFormat::kInt16)
              ? MTL::IndexTypeUInt16
              : MTL::IndexTypeUInt32;
      MTL::Buffer* index_buffer = nullptr;
      uint64_t index_offset = 0;
      if (!resolve_index_buffer(index_type, index_buffer, index_offset)) {
        return false;
      }
      IRRuntimeDrawIndexedPrimitives(
          current_render_encoder_, mtl_primitive,
          NS::UInteger(primitive_processing_result.host_draw_vertex_count),
          index_type, index_buffer, index_offset, NS::UInteger(1), 0, 0);
    }
  }

  if (memexport_used && shared_memory_) {
    for (const draw_util::MemExportRange& memexport_range : memexport_ranges_) {
      shared_memory_->RangeWrittenByGpu(
          memexport_range.base_address_dwords << 2, memexport_range.size_bytes);
    }
  }

  submission_has_draws_ = true;
  return true;
}

MTL::CommandBuffer* MetalCommandProcessor::BeginResolveOrdering() {
  // End any in-flight rendering so render target contents are visible to
  // resolve logic.
  EndRenderEncoder();

  switch (resolve_ordering_policy_) {
    case ResolveOrderingPolicy::kSubmissionBoundary: {
      bool had_prior_draws = submission_has_draws_;
      if (had_prior_draws) {
        // D3D12 can keep resolves in the same submission because it tracks
        // shared memory / scaled-resolve UAV write visibility explicitly.
        // The Metal path doesn't have an equivalent state machine yet, so
        // keep draw payload and resolve payload in separate command buffers
        // for correctness.
        EndCommandBuffer();
      }
      break;
    }
  }

  return EnsureCommandBuffer();
}

void MetalCommandProcessor::EndResolveOrdering() {
  switch (resolve_ordering_policy_) {
    case ResolveOrderingPolicy::kSubmissionBoundary:
      // Until Metal has a real D3D12-style pending-write state machine,
      // resolved data must become visible at a command-buffer boundary
      // immediately rather than being left pending into a later draw or
      // wait boundary.
      EndCommandBuffer();
      break;
  }
}

bool MetalCommandProcessor::IssueCopy() {
  // ===========================================================================
  // Host render backend copy/resolve entry point.
  //
  // The virtual BeginResolveOrdering / EndResolveOrdering pair bracket the
  // resolve work and enforce the active ResolveOrderingPolicy (currently
  // kSubmissionBoundary).  The actual resolve is delegated to
  // MetalRenderTargetCache::Resolve.  A future strict backend can override
  // the ordering methods to use on-tile resolve without separate submissions.
  // ===========================================================================

  MTL::CommandBuffer* copy_command_buffer = BeginResolveOrdering();
  if (!copy_command_buffer) {
    XELOGE("MetalCommandProcessor::IssueCopy: failed to get command buffer");
    return false;
  }

  if (!render_target_cache_) {
    XELOGW("MetalCommandProcessor::IssueCopy - No render target cache");
    return true;
  }

  uint32_t written_address = 0;
  uint32_t written_length = 0;

  if (!render_target_cache_->Resolve(*memory_, written_address, written_length,
                                     copy_command_buffer)) {
    XELOGE("MetalCommandProcessor::IssueCopy - Resolve failed");
    return false;
  }

  if (!written_length) {
    return true;
  }

  // Track this resolved region so the trace player can avoid overwriting it
  // with stale MemoryRead commands from the trace file.
  trace_resolve_guard_.Mark(written_address, written_length);

  EndResolveOrdering();
  return true;
}

void MetalCommandProcessor::OnGammaRamp256EntryTableValueWritten() {
  gamma_ramp_256_entry_table_up_to_date_ = false;
}

void MetalCommandProcessor::OnGammaRampPWLValueWritten() {
  gamma_ramp_pwl_up_to_date_ = false;
}

void MetalCommandProcessor::WriteRegister(uint32_t index, uint32_t value) {
  CommandProcessor::WriteRegister(index, value);

  if (index >= XE_GPU_REG_SHADER_CONSTANT_000_X &&
      index <= XE_GPU_REG_SHADER_CONSTANT_511_W) {
    uint32_t float_constant_index =
        (index - XE_GPU_REG_SHADER_CONSTANT_000_X) >> 2;
    if (float_constant_index >= 256) {
      uint32_t rel = float_constant_index & 0xFF;
      if (current_float_constant_map_pixel_[rel >> 6] &
          (uint64_t(1) << (rel & 63))) {
        cbuffer_binding_float_pixel_.up_to_date = false;
      }
    } else {
      if (current_float_constant_map_vertex_[float_constant_index >> 6] &
          (uint64_t(1) << (float_constant_index & 63))) {
        cbuffer_binding_float_vertex_.up_to_date = false;
      }
    }
  } else if (index >= XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031 &&
             index <= XE_GPU_REG_SHADER_CONSTANT_LOOP_31) {
    cbuffer_binding_bool_loop_.up_to_date = false;
  } else if (index >= XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 &&
             index <= XE_GPU_REG_SHADER_CONSTANT_FETCH_31_5) {
    cbuffer_binding_fetch_.up_to_date = false;
    if (texture_cache_) {
      texture_cache_->TextureFetchConstantWritten(
          (index - XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0) / 6);
    }
  }
}

MTL::CommandBuffer* MetalCommandProcessor::EnsureCommandBuffer() {
  ProcessCompletedSubmissions();
  if (current_command_buffer_) {
    return current_command_buffer_;
  }
  if (!command_queue_) {
    XELOGE("EnsureCommandBuffer: no command queue");
    return nullptr;
  }

  EnsureCommandBufferAutoreleasePool();

  // Note: commandBuffer() returns an autoreleased object, we must retain it.
  current_command_buffer_ = command_queue_->commandBuffer();
  if (!current_command_buffer_) {
    XELOGE("EnsureCommandBuffer: failed to create command buffer");
    DrainCommandBufferAutoreleasePool();
    return nullptr;
  }
  current_command_buffer_->retain();

  ++submission_current_;
  current_command_buffer_->setLabel(
      NS::String::string("XeniaCommandBuffer", NS::UTF8StringEncoding));

  pending_completion_handlers_.fetch_add(1, std::memory_order_relaxed);
  current_command_buffer_->addCompletedHandler(
      [this](MTL::CommandBuffer* completed_cmd) {
        if (completed_cmd->status() == MTL::CommandBufferStatusError) {
          NS::Error* error = completed_cmd->error();
          if (error) {
            XELOGE("Metal command buffer error: {}",
                   error->localizedDescription()->utf8String());
          }
        }
        completed_command_buffers_.fetch_add(1, std::memory_order_relaxed);
        pending_completion_handlers_.fetch_sub(1, std::memory_order_relaxed);
      });

  if (texture_cache_) {
    texture_cache_->BeginSubmission(submission_current_);
  }
  submission_has_draws_ = false;
  if (primitive_processor_ && !frame_open_) {
    primitive_processor_->BeginFrame();
    if (render_target_cache_) {
      render_target_cache_->BeginFrame();
    }
    if (texture_cache_) {
      texture_cache_->BeginFrame();
    }
    frame_open_ = true;
  }

  return current_command_buffer_;
}

void MetalCommandProcessor::ProcessCompletedSubmissions() {
  const uint64_t completed =
      completed_command_buffers_.load(std::memory_order_relaxed);
  if (completed <= submission_completed_processed_) {
    return;
  }
  submission_completed_processed_ = completed;
  if (constant_buffer_pool_) {
    constant_buffer_pool_->Reclaim(completed);
  }
  if (texture_cache_) {
    texture_cache_->CompletedSubmissionUpdated(completed);
  }
  while (!retired_view_bindless_indices_.empty() &&
         retired_view_bindless_indices_.front().submission_id <= completed) {
    FreeViewBindlessIndexNow(retired_view_bindless_indices_.front().index);
    retired_view_bindless_indices_.pop_front();
  }
  while (!retired_sampler_bindless_indices_.empty() &&
         retired_sampler_bindless_indices_.front().submission_id <= completed) {
    FreeSamplerBindlessIndexNow(
        retired_sampler_bindless_indices_.front().index);
    retired_sampler_bindless_indices_.pop_front();
  }
}

void MetalCommandProcessor::EnsureCommandBufferAutoreleasePool() {
  if (command_buffer_autorelease_pool_) {
    return;
  }
  command_buffer_autorelease_pool_ = NS::AutoreleasePool::alloc()->init();
}

void MetalCommandProcessor::DrainCommandBufferAutoreleasePool() {
  if (!command_buffer_autorelease_pool_) {
    return;
  }
  command_buffer_autorelease_pool_->release();
  command_buffer_autorelease_pool_ = nullptr;
}

void MetalCommandProcessor::EndRenderEncoder() {
  if (!current_render_encoder_) {
    return;
  }
  current_render_encoder_->endEncoding();
  current_render_encoder_->release();
  current_render_encoder_ = nullptr;
  current_render_pass_descriptor_ = nullptr;
  current_render_pipeline_state_ = nullptr;
  rasterizer_state_valid_ = false;
  current_depth_stencil_state_ = nullptr;
  stencil_reference_valid_ = false;
  heap_binds_set_on_encoder_ = false;
  current_bindless_table_valid_ = false;
}

MTL::CommandBuffer* MetalCommandProcessor::RequestTransferCommandBuffer() {
  EndRenderEncoder();
  return EnsureCommandBuffer();
}

MTL::CommandBuffer*
MetalCommandProcessor::CreateStandaloneTransferCommandBuffer(
    const char* label) {
  if (!command_queue_) {
    return nullptr;
  }
  MTL::CommandBuffer* cmd = command_queue_->commandBuffer();
  if (!cmd) {
    return nullptr;
  }
  cmd->retain();
  (void)label;
  return cmd;
}

void MetalCommandProcessor::CommitStandaloneAsync(MTL::CommandBuffer* cmd) {
  if (!cmd) {
    return;
  }
  cmd->addCompletedHandler(^(MTL::CommandBuffer* completed_cmd) {
    completed_cmd->release();
  });
  cmd->commit();
}

void MetalCommandProcessor::CommitStandaloneAndWait(MTL::CommandBuffer* cmd) {
  if (!cmd) {
    return;
  }
  cmd->commit();
  cmd->waitUntilCompleted();
  cmd->release();
}

void MetalCommandProcessor::ResetRenderEncoderResourceUsage() {
  render_encoder_resource_usage_.clear();
  render_encoder_heap_usage_.clear();
}

void MetalCommandProcessor::UseRenderEncoderResource(MTL::Resource* resource,
                                                     MTL::ResourceUsage usage) {
  if (!current_render_encoder_ || !resource) {
    return;
  }
  UseRenderEncoderHeap(resource->heap());
  uint32_t usage_bits = static_cast<uint32_t>(usage);
  for (auto& resource_usage : render_encoder_resource_usage_) {
    if (resource_usage.resource != resource) {
      continue;
    }
    if ((resource_usage.usage_bits & usage_bits) == usage_bits) {
      return;
    }
    resource_usage.usage_bits |= usage_bits;
    current_render_encoder_->useResource(resource, usage);
    return;
  }
  render_encoder_resource_usage_.push_back({resource, usage_bits});
  current_render_encoder_->useResource(resource, usage);
}

void MetalCommandProcessor::UseRenderEncoderHeap(MTL::Heap* heap) {
  if (!current_render_encoder_ || !heap) {
    return;
  }
  for (MTL::Heap* used_heap : render_encoder_heap_usage_) {
    if (used_heap == heap) {
      return;
    }
  }
  render_encoder_heap_usage_.push_back(heap);
  current_render_encoder_->useHeap(heap);
}

void MetalCommandProcessor::UseRenderEncoderAttachmentHeaps(
    MTL::RenderPassDescriptor* descriptor) {
  if (!current_render_encoder_ || !descriptor) {
    return;
  }
  auto* color_attachments = descriptor->colorAttachments();
  for (uint32_t i = 0; i < 8; ++i) {
    auto* attachment = color_attachments->object(i);
    if (!attachment) {
      continue;
    }
    MTL::Texture* texture = attachment->texture();
    if (texture) {
      UseRenderEncoderHeap(texture->heap());
    }
  }
  auto* depth_attachment = descriptor->depthAttachment();
  if (depth_attachment && depth_attachment->texture()) {
    UseRenderEncoderHeap(depth_attachment->texture()->heap());
  }
  auto* stencil_attachment = descriptor->stencilAttachment();
  if (stencil_attachment && stencil_attachment->texture()) {
    UseRenderEncoderHeap(stencil_attachment->texture()->heap());
  }
}

void MetalCommandProcessor::BeginCommandBuffer() {
  if (!EnsureCommandBuffer()) {
    return;
  }

  if (!current_render_encoder_ && (!render_encoder_resource_usage_.empty() ||
                                   !render_encoder_heap_usage_.empty())) {
    ResetRenderEncoderResourceUsage();
  }

  // Obtain the render pass descriptor. Prefer the one provided by
  // MetalRenderTargetCache (host render-target path), falling back to the
  // legacy descriptor if needed.
  MTL::RenderPassDescriptor* pass_descriptor = nullptr;
  if (render_target_cache_) {
    if (MTL::RenderPassDescriptor* cache_desc =
            render_target_cache_->GetRenderPassDescriptor(1)) {
      pass_descriptor = cache_desc;
    }
  }
  if (!pass_descriptor) {
    XELOGE("BeginCommandBuffer: No render pass descriptor available");
    return;
  }

  // Detect Reverse-Z usage and update clear depth.
  if (register_file_) {
    auto depth_control = register_file_->Get<reg::RB_DEPTHCONTROL>();
    bool reverse_z =
        depth_control.z_enable &&
        (depth_control.zfunc == xenos::CompareFunction::kGreater ||
         depth_control.zfunc == xenos::CompareFunction::kGreaterEqual);
    if (auto* da = pass_descriptor->depthAttachment()) {
      if (reverse_z) {
        da->setClearDepth(0.0);
      } else {
        da->setClearDepth(1.0);
      }
    }
  }

  // If the render pass configuration has changed since the current render
  // encoder was created (e.g. dummy RT0 -> real RTs, depth/stencil binding),
  // restart the render encoder with the updated descriptor.
  if (current_render_encoder_ &&
      current_render_pass_descriptor_ != pass_descriptor) {
    EndRenderEncoder();
  }

  if (!current_render_encoder_) {
    // If some path cleared the encoder without going through EndRenderEncoder,
    // avoid leaking cached binding state into the new encoder.
    // Note: renderCommandEncoder() returns an autoreleased object, we must
    // retain it.
    current_render_encoder_ =
        current_command_buffer_->renderCommandEncoder(pass_descriptor);
    if (!current_render_encoder_) {
      XELOGE("Failed to create render command encoder");
      return;
    }
    current_render_encoder_->retain();
    current_render_encoder_->setLabel(
        NS::String::string("XeniaRenderEncoder", NS::UTF8StringEncoding));
    current_render_pipeline_state_ = nullptr;
    ff_blend_factor_valid_ = false;
    rasterizer_state_valid_ = false;
    viewport_dirty_ = true;
    scissor_dirty_ = true;
    current_depth_stencil_state_ = nullptr;
    stencil_reference_valid_ = false;
    heap_binds_set_on_encoder_ = false;
    current_render_pass_descriptor_ = pass_descriptor;
    UseRenderEncoderAttachmentHeaps(pass_descriptor);
  }

  // Derive viewport/scissor from the actual bound render target rather than
  // a hard-coded 1280x720. Prefer color RT 0 from the MetalRenderTargetCache,
  // falling back to depth (depth-only passes) and then 1280x720.
  uint32_t rt_width = 1;
  uint32_t rt_height = 1;
  GetBoundRenderTargetSize(render_target_cache_.get(), 1280, 720, rt_width,
                           rt_height);

  // Set viewport
  MTL::Viewport viewport = {
      0.0, 0.0, static_cast<double>(rt_width), static_cast<double>(rt_height),
      0.0, 1.0};
  current_render_encoder_->setViewport(viewport);

  // Set scissor (must not exceed render pass dimensions)
  MTL::ScissorRect scissor = {0, 0, rt_width, rt_height};
  current_render_encoder_->setScissorRect(scissor);

  // Mark dirty so IssueDraw re-applies the per-draw viewport/scissor.
  viewport_dirty_ = true;
  scissor_dirty_ = true;
}

void MetalCommandProcessor::EndCommandBuffer() {
  EndRenderEncoder();

  if (current_command_buffer_) {
    current_command_buffer_->commit();
    current_command_buffer_->release();
    current_command_buffer_ = nullptr;
    submission_has_draws_ = false;
    current_bindless_table_valid_ = false;
  }
  copy_resolve_writes_pending_ = false;
  DrainCommandBufferAutoreleasePool();
}

void MetalCommandProcessor::ApplyDepthStencilState(
    bool primitive_polygonal, reg::RB_DEPTHCONTROL normalized_depth_control) {
  if (!current_render_encoder_ || !device_) {
    return;
  }

  const RegisterFile& regs = *register_file_;
  auto stencil_ref_mask_front = regs.Get<reg::RB_STENCILREFMASK>();
  auto stencil_ref_mask_back =
      regs.Get<reg::RB_STENCILREFMASK>(XE_GPU_REG_RB_STENCILREFMASK_BF);
  auto depth_control = normalized_depth_control;

  bool has_stencil_attachment = false;
  if (current_render_pass_descriptor_) {
    if (auto* stencil_attachment =
            current_render_pass_descriptor_->stencilAttachment()) {
      has_stencil_attachment = stencil_attachment->texture() != nullptr;
    }
  }

  if (!has_stencil_attachment && depth_control.stencil_enable) {
    static bool no_stencil_logged = false;
    if (!no_stencil_logged) {
      no_stencil_logged = true;
      XELOGW(
          "Metal: stencil enabled but no stencil attachment bound; disabling "
          "stencil for this pass");
    }
    depth_control.stencil_enable = 0;
    depth_control.backface_enable = 0;
    depth_control.stencilfunc = xenos::CompareFunction::kAlways;
    depth_control.stencilfail = xenos::StencilOp::kKeep;
    depth_control.stencilzpass = xenos::StencilOp::kKeep;
    depth_control.stencilzfail = xenos::StencilOp::kKeep;
    depth_control.stencilfunc_bf = xenos::CompareFunction::kAlways;
    depth_control.stencilfail_bf = xenos::StencilOp::kKeep;
    depth_control.stencilzpass_bf = xenos::StencilOp::kKeep;
    depth_control.stencilzfail_bf = xenos::StencilOp::kKeep;
    stencil_ref_mask_front.value = 0;
    stencil_ref_mask_back.value = 0;
  }

  DepthStencilStateKey key;
  key.depth_control = depth_control.value;
  key.stencil_ref_mask_front = stencil_ref_mask_front.value;
  key.stencil_ref_mask_back = stencil_ref_mask_back.value;
  key.polygonal_and_backface = (primitive_polygonal ? 1u : 0u) |
                               (depth_control.backface_enable ? 2u : 0u);

  MTL::DepthStencilState* state = nullptr;
  auto it = depth_stencil_state_cache_.find(key);
  if (it != depth_stencil_state_cache_.end()) {
    state = it->second;
  } else {
    MTL::DepthStencilDescriptor* ds_desc =
        MTL::DepthStencilDescriptor::alloc()->init();
    if (depth_control.z_enable) {
      ds_desc->setDepthCompareFunction(
          ToMetalCompareFunction(depth_control.zfunc));
      ds_desc->setDepthWriteEnabled(depth_control.z_write_enable != 0);
    } else {
      ds_desc->setDepthCompareFunction(MTL::CompareFunctionAlways);
      ds_desc->setDepthWriteEnabled(false);
    }

    if (depth_control.stencil_enable) {
      auto* front = MTL::StencilDescriptor::alloc()->init();
      front->setStencilCompareFunction(
          ToMetalCompareFunction(depth_control.stencilfunc));
      front->setStencilFailureOperation(
          ToMetalStencilOperation(depth_control.stencilfail));
      front->setDepthFailureOperation(
          ToMetalStencilOperation(depth_control.stencilzfail));
      front->setDepthStencilPassOperation(
          ToMetalStencilOperation(depth_control.stencilzpass));
      front->setReadMask(stencil_ref_mask_front.stencilmask);
      front->setWriteMask(stencil_ref_mask_front.stencilwritemask);

      ds_desc->setFrontFaceStencil(front);

      if (primitive_polygonal && depth_control.backface_enable) {
        auto* back = MTL::StencilDescriptor::alloc()->init();
        back->setStencilCompareFunction(
            ToMetalCompareFunction(depth_control.stencilfunc_bf));
        back->setStencilFailureOperation(
            ToMetalStencilOperation(depth_control.stencilfail_bf));
        back->setDepthFailureOperation(
            ToMetalStencilOperation(depth_control.stencilzfail_bf));
        back->setDepthStencilPassOperation(
            ToMetalStencilOperation(depth_control.stencilzpass_bf));
        back->setReadMask(stencil_ref_mask_back.stencilmask);
        back->setWriteMask(stencil_ref_mask_back.stencilwritemask);
        ds_desc->setBackFaceStencil(back);
        back->release();
      } else {
        ds_desc->setBackFaceStencil(front);
      }

      front->release();
    }

    state = device_->newDepthStencilState(ds_desc);
    ds_desc->release();

    if (!state) {
      XELOGE("Failed to create Metal depth/stencil state");
      return;
    }
    depth_stencil_state_cache_.emplace(key, state);
  }

  if (current_depth_stencil_state_ != state) {
    current_render_encoder_->setDepthStencilState(state);
    current_depth_stencil_state_ = state;
  }

  if (depth_control.stencil_enable) {
    uint32_t ref_front = stencil_ref_mask_front.stencilref;
    uint32_t ref_back = stencil_ref_mask_back.stencilref;
    auto pa_su_sc_mode_cntl = regs.Get<reg::PA_SU_SC_MODE_CNTL>();
    uint32_t ref = ref_front;
    if (primitive_polygonal && depth_control.backface_enable &&
        pa_su_sc_mode_cntl.cull_front && !pa_su_sc_mode_cntl.cull_back) {
      ref = ref_back;
    } else if (primitive_polygonal && depth_control.backface_enable &&
               ref_front != ref_back) {
      static bool mismatch_logged = false;
      if (!mismatch_logged) {
        mismatch_logged = true;
        XELOGW(
            "Metal: front/back stencil ref differ (front={}, back={}); using "
            "front for both",
            ref_front, ref_back);
      }
    }
    if (!stencil_reference_valid_ || current_stencil_reference_ != ref) {
      current_render_encoder_->setStencilReferenceValue(ref);
      current_stencil_reference_ = ref;
      stencil_reference_valid_ = true;
    }
  }
}

void MetalCommandProcessor::ApplyRasterizerState(bool primitive_polygonal) {
  if (!current_render_encoder_ || !render_target_cache_) {
    return;
  }

  const RegisterFile& regs = *register_file_;
  auto pa_su_sc_mode_cntl = regs.Get<reg::PA_SU_SC_MODE_CNTL>();
  auto pa_cl_clip_cntl = regs.Get<reg::PA_CL_CLIP_CNTL>();

  MTL::CullMode cull_mode = MTL::CullModeNone;
  if (primitive_polygonal) {
    bool cull_front = pa_su_sc_mode_cntl.cull_front;
    bool cull_back = pa_su_sc_mode_cntl.cull_back;
    if (cull_front && !cull_back) {
      cull_mode = MTL::CullModeFront;
    } else if (cull_back && !cull_front) {
      cull_mode = MTL::CullModeBack;
    }
  }
  if (!rasterizer_state_valid_ || current_cull_mode_ != cull_mode) {
    current_render_encoder_->setCullMode(cull_mode);
    current_cull_mode_ = cull_mode;
  }

  MTL::Winding front_facing_winding = pa_su_sc_mode_cntl.face
                                          ? MTL::WindingClockwise
                                          : MTL::WindingCounterClockwise;
  if (!rasterizer_state_valid_ ||
      current_front_facing_winding_ != front_facing_winding) {
    current_render_encoder_->setFrontFacingWinding(front_facing_winding);
    current_front_facing_winding_ = front_facing_winding;
  }

  MTL::TriangleFillMode fill_mode = MTL::TriangleFillModeFill;
  if (primitive_polygonal &&
      pa_su_sc_mode_cntl.poly_mode == xenos::PolygonModeEnable::kDualMode) {
    xenos::PolygonType polygon_type = xenos::PolygonType::kTriangles;
    if (!pa_su_sc_mode_cntl.cull_front) {
      polygon_type =
          std::min(polygon_type, pa_su_sc_mode_cntl.polymode_front_ptype);
    }
    if (!pa_su_sc_mode_cntl.cull_back) {
      polygon_type =
          std::min(polygon_type, pa_su_sc_mode_cntl.polymode_back_ptype);
    }
    if (polygon_type != xenos::PolygonType::kTriangles) {
      fill_mode = MTL::TriangleFillModeLines;
    }
  }
  if (!rasterizer_state_valid_ || current_triangle_fill_mode_ != fill_mode) {
    current_render_encoder_->setTriangleFillMode(fill_mode);
    current_triangle_fill_mode_ = fill_mode;
  }

  float polygon_offset_scale = 0.0f;
  float polygon_offset = 0.0f;
  draw_util::GetPreferredFacePolygonOffset(
      regs, primitive_polygonal, polygon_offset_scale, polygon_offset);
  float depth_bias_constant =
      static_cast<float>(draw_util::GetD3D10IntegerPolygonOffset(
          regs.Get<reg::RB_DEPTH_INFO>().depth_format, polygon_offset));
  float depth_bias_slope =
      polygon_offset_scale * xenos::kPolygonOffsetScaleSubpixelUnit *
      float(std::max(render_target_cache_->draw_resolution_scale_x(),
                     render_target_cache_->draw_resolution_scale_y()));
  float depth_bias_values[] = {depth_bias_constant, depth_bias_slope, 0.0f};
  if (!rasterizer_state_valid_ ||
      std::memcmp(current_depth_bias_values_, depth_bias_values,
                  sizeof(depth_bias_values)) != 0) {
    current_render_encoder_->setDepthBias(depth_bias_constant, depth_bias_slope,
                                          0.0f);
    std::memcpy(current_depth_bias_values_, depth_bias_values,
                sizeof(depth_bias_values));
  }

  MTL::DepthClipMode depth_clip_mode = pa_cl_clip_cntl.clip_disable
                                           ? MTL::DepthClipModeClamp
                                           : MTL::DepthClipModeClip;
  if (!rasterizer_state_valid_ || current_depth_clip_mode_ != depth_clip_mode) {
    current_render_encoder_->setDepthClipMode(depth_clip_mode);
    current_depth_clip_mode_ = depth_clip_mode;
  }
  rasterizer_state_valid_ = true;
}

void MetalCommandProcessor::UpdateSystemConstantValues(
    bool shared_memory_is_uav, bool primitive_polygonal,
    uint32_t line_loop_closing_index, xenos::Endian index_endian,
    const draw_util::ViewportInfo& viewport_info, uint32_t used_texture_mask,
    reg::RB_DEPTHCONTROL normalized_depth_control,
    uint32_t normalized_color_mask) {
  const RegisterFile& regs = *register_file_;
  auto pa_cl_clip_cntl = regs.Get<reg::PA_CL_CLIP_CNTL>();
  auto pa_cl_vte_cntl = regs.Get<reg::PA_CL_VTE_CNTL>();
  auto rb_alpha_ref = regs.Get<float>(XE_GPU_REG_RB_ALPHA_REF);
  auto rb_colorcontrol = regs.Get<reg::RB_COLORCONTROL>();
  auto rb_depth_info = regs.Get<reg::RB_DEPTH_INFO>();
  auto rb_surface_info = regs.Get<reg::RB_SURFACE_INFO>();
  auto vgt_draw_initiator = regs.Get<reg::VGT_DRAW_INITIATOR>();
  uint32_t vgt_indx_offset = regs.Get<reg::VGT_INDX_OFFSET>().indx_offset;
  uint32_t vgt_max_vtx_indx = regs.Get<reg::VGT_MAX_VTX_INDX>().max_indx;
  uint32_t vgt_min_vtx_indx = regs.Get<reg::VGT_MIN_VTX_INDX>().min_indx;

  uint32_t dirty = 0u;
  ArchFloatMask dirty_float_mask = floatmask_zero;

  auto update_dirty_floatmask = [&dirty_float_mask](float x, float y) {
    dirty_float_mask =
        ArchORFloatMask(dirty_float_mask, ArchCmpneqFloatMask(x, y));
  };
  auto update_dirty_uint32_cmp = [&dirty](uint32_t x, uint32_t y) {
    dirty |= (x ^ y);
  };

  // Get color info for each render target
  reg::RB_COLOR_INFO color_infos[4];
  for (uint32_t i = 0; i < 4; ++i) {
    color_infos[i] = regs.Get<reg::RB_COLOR_INFO>(
        reg::RB_COLOR_INFO::rt_register_indices[i]);
  }

  // Build flags
  uint32_t flags = 0;

  // Shared memory mode - determines whether shaders read from SRV (T0) or UAV
  // (U0)
  if (shared_memory_is_uav) {
    flags |= DxbcShaderTranslator::kSysFlag_SharedMemoryIsUAV;
  }

  // W0 division control from PA_CL_VTE_CNTL
  if (pa_cl_vte_cntl.vtx_xy_fmt) {
    flags |= DxbcShaderTranslator::kSysFlag_XYDividedByW;
  }
  if (pa_cl_vte_cntl.vtx_z_fmt) {
    flags |= DxbcShaderTranslator::kSysFlag_ZDividedByW;
  }
  if (pa_cl_vte_cntl.vtx_w0_fmt) {
    flags |= DxbcShaderTranslator::kSysFlag_WNotReciprocal;
  }

  // Primitive type flags
  if (primitive_polygonal) {
    flags |= DxbcShaderTranslator::kSysFlag_PrimitivePolygonal;
  }
  if (draw_util::IsPrimitiveLine(regs)) {
    flags |= DxbcShaderTranslator::kSysFlag_PrimitiveLine;
  }

  // Depth format
  if (rb_depth_info.depth_format == xenos::DepthRenderTargetFormat::kD24FS8) {
    flags |= DxbcShaderTranslator::kSysFlag_DepthFloat24;
  }

  // Alpha test - encode compare function in flags
  xenos::CompareFunction alpha_test_function =
      rb_colorcontrol.alpha_test_enable ? rb_colorcontrol.alpha_func
                                        : xenos::CompareFunction::kAlways;
  flags |= uint32_t(alpha_test_function)
           << DxbcShaderTranslator::kSysFlag_AlphaPassIfLess_Shift;

  // Gamma conversion flags for render targets
  if (!render_target_cache_->gamma_render_target_as_unorm16()) {
    for (uint32_t i = 0; i < 4; ++i) {
      if (color_infos[i].color_format ==
          xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA) {
        flags |= DxbcShaderTranslator::kSysFlag_ConvertColor0ToGamma << i;
      }
    }
  }

  update_dirty_uint32_cmp(system_constants_.flags, flags);
  system_constants_.flags = flags;

  // Tessellation factor range
  float tessellation_factor_min =
      regs.Get<float>(XE_GPU_REG_VGT_HOS_MIN_TESS_LEVEL) + 1.0f;
  float tessellation_factor_max =
      regs.Get<float>(XE_GPU_REG_VGT_HOS_MAX_TESS_LEVEL) + 1.0f;
  update_dirty_floatmask(system_constants_.tessellation_factor_range_min,
                         tessellation_factor_min);
  update_dirty_floatmask(system_constants_.tessellation_factor_range_max,
                         tessellation_factor_max);
  system_constants_.tessellation_factor_range_min = tessellation_factor_min;
  system_constants_.tessellation_factor_range_max = tessellation_factor_max;

  // Line loop closing index
  update_dirty_uint32_cmp(system_constants_.line_loop_closing_index,
                          line_loop_closing_index);
  system_constants_.line_loop_closing_index = line_loop_closing_index;

  // Vertex index configuration
  update_dirty_uint32_cmp(
      static_cast<uint32_t>(system_constants_.vertex_index_endian),
      static_cast<uint32_t>(index_endian));
  update_dirty_uint32_cmp(system_constants_.vertex_index_offset,
                          vgt_indx_offset);
  update_dirty_uint32_cmp(system_constants_.vertex_index_min, vgt_min_vtx_indx);
  update_dirty_uint32_cmp(system_constants_.vertex_index_max, vgt_max_vtx_indx);
  system_constants_.vertex_index_endian = index_endian;
  system_constants_.vertex_index_offset = vgt_indx_offset;
  system_constants_.vertex_index_min = vgt_min_vtx_indx;
  system_constants_.vertex_index_max = vgt_max_vtx_indx;

  // User clip planes (when not CLIP_DISABLE)
  if (!pa_cl_clip_cntl.clip_disable) {
    float* user_clip_plane_write_ptr = system_constants_.user_clip_planes[0];
    uint32_t user_clip_planes_remaining = pa_cl_clip_cntl.ucp_ena;
    uint32_t user_clip_plane_index;
    while (xe::bit_scan_forward(user_clip_planes_remaining,
                                &user_clip_plane_index)) {
      user_clip_planes_remaining &= ~(UINT32_C(1) << user_clip_plane_index);
      const float* user_clip_plane_regs = reinterpret_cast<const float*>(
          &regs.values[XE_GPU_REG_PA_CL_UCP_0_X + user_clip_plane_index * 4]);
      if (std::memcmp(user_clip_plane_write_ptr, user_clip_plane_regs,
                      4 * sizeof(float)) != 0) {
        dirty = true;
        std::memcpy(user_clip_plane_write_ptr, user_clip_plane_regs,
                    4 * sizeof(float));
      }
      user_clip_plane_write_ptr += 4;
    }
  }

  // NDC scale and offset from viewport info
  for (uint32_t i = 0; i < 3; ++i) {
    update_dirty_floatmask(system_constants_.ndc_scale[i],
                           viewport_info.ndc_scale[i]);
    update_dirty_floatmask(system_constants_.ndc_offset[i],
                           viewport_info.ndc_offset[i]);
    system_constants_.ndc_scale[i] = viewport_info.ndc_scale[i];
    system_constants_.ndc_offset[i] = viewport_info.ndc_offset[i];
  }

  // Point size parameters
  if (vgt_draw_initiator.prim_type == xenos::PrimitiveType::kPointList) {
    auto pa_su_point_minmax = regs.Get<reg::PA_SU_POINT_MINMAX>();
    auto pa_su_point_size = regs.Get<reg::PA_SU_POINT_SIZE>();
    float point_vertex_diameter_min =
        float(pa_su_point_minmax.min_size) * (2.0f / 16.0f);
    float point_vertex_diameter_max =
        float(pa_su_point_minmax.max_size) * (2.0f / 16.0f);
    float point_constant_diameter_x =
        float(pa_su_point_size.width) * (2.0f / 16.0f);
    float point_constant_diameter_y =
        float(pa_su_point_size.height) * (2.0f / 16.0f);
    update_dirty_floatmask(system_constants_.point_vertex_diameter_min,
                           point_vertex_diameter_min);
    update_dirty_floatmask(system_constants_.point_vertex_diameter_max,
                           point_vertex_diameter_max);
    update_dirty_floatmask(system_constants_.point_constant_diameter[0],
                           point_constant_diameter_x);
    update_dirty_floatmask(system_constants_.point_constant_diameter[1],
                           point_constant_diameter_y);
    system_constants_.point_vertex_diameter_min = point_vertex_diameter_min;
    system_constants_.point_vertex_diameter_max = point_vertex_diameter_max;
    system_constants_.point_constant_diameter[0] = point_constant_diameter_x;
    system_constants_.point_constant_diameter[1] = point_constant_diameter_y;
    // Screen to NDC radius conversion.
    // 2 because 1 in the NDC is half of the viewport's axis, 0.5 for diameter
    // to radius conversion to avoid multiplying the per-vertex diameter by an
    // additional constant in the shader. Include draw_resolution_scale to
    // match D3D12 behavior.
    uint32_t point_draw_resolution_scale_x =
        render_target_cache_ ? render_target_cache_->draw_resolution_scale_x()
                             : 1;
    uint32_t point_draw_resolution_scale_y =
        render_target_cache_ ? render_target_cache_->draw_resolution_scale_y()
                             : 1;
    float point_screen_diameter_to_ndc_radius_x =
        (/* 0.5f * 2.0f * */ float(point_draw_resolution_scale_x)) /
        std::max(viewport_info.xy_extent[0], uint32_t(1));
    float point_screen_diameter_to_ndc_radius_y =
        (/* 0.5f * 2.0f * */ float(point_draw_resolution_scale_y)) /
        std::max(viewport_info.xy_extent[1], uint32_t(1));
    update_dirty_floatmask(
        system_constants_.point_screen_diameter_to_ndc_radius[0],
        point_screen_diameter_to_ndc_radius_x);
    update_dirty_floatmask(
        system_constants_.point_screen_diameter_to_ndc_radius[1],
        point_screen_diameter_to_ndc_radius_y);
    system_constants_.point_screen_diameter_to_ndc_radius[0] =
        point_screen_diameter_to_ndc_radius_x;
    system_constants_.point_screen_diameter_to_ndc_radius[1] =
        point_screen_diameter_to_ndc_radius_y;
  }

  // Texture signedness / resolution scaling (mirror D3D12 logic).
  // Always update textures_resolution_scaled, even when used_texture_mask is 0,
  // to avoid stale values from previous draws.
  uint32_t textures_resolution_scaled = 0;
  uint32_t textures_remaining = used_texture_mask;
  uint32_t texture_index;
  while (xe::bit_scan_forward(textures_remaining, &texture_index)) {
    textures_remaining &= ~(uint32_t(1) << texture_index);
    if (texture_cache_) {
      uint32_t& texture_signs_uint =
          system_constants_.texture_swizzled_signs[texture_index >> 2];
      uint32_t texture_signs_shift = (texture_index & 3) * 8;
      uint8_t texture_signs =
          texture_cache_->GetActiveTextureSwizzledSigns(texture_index);
      uint32_t texture_signs_shifted = uint32_t(texture_signs)
                                       << texture_signs_shift;
      uint32_t texture_signs_mask = uint32_t(0xFF) << texture_signs_shift;
      update_dirty_uint32_cmp((texture_signs_uint & texture_signs_mask),
                              texture_signs_shifted);
      texture_signs_uint =
          (texture_signs_uint & ~texture_signs_mask) | texture_signs_shifted;
      textures_resolution_scaled |=
          uint32_t(
              texture_cache_->IsActiveTextureResolutionScaled(texture_index))
          << texture_index;
    }
  }
  update_dirty_uint32_cmp(system_constants_.textures_resolution_scaled,
                          textures_resolution_scaled);
  system_constants_.textures_resolution_scaled = textures_resolution_scaled;

  // Sample count log2 for alpha to mask
  uint32_t sample_count_log2_x =
      rb_surface_info.msaa_samples >= xenos::MsaaSamples::k4X ? 1 : 0;
  uint32_t sample_count_log2_y =
      rb_surface_info.msaa_samples >= xenos::MsaaSamples::k2X ? 1 : 0;
  update_dirty_uint32_cmp(system_constants_.sample_count_log2[0],
                          sample_count_log2_x);
  update_dirty_uint32_cmp(system_constants_.sample_count_log2[1],
                          sample_count_log2_y);
  system_constants_.sample_count_log2[0] = sample_count_log2_x;
  system_constants_.sample_count_log2[1] = sample_count_log2_y;

  // Alpha test reference
  update_dirty_floatmask(system_constants_.alpha_test_reference, rb_alpha_ref);
  system_constants_.alpha_test_reference = rb_alpha_ref;

  // Alpha to mask
  uint32_t alpha_to_mask = rb_colorcontrol.alpha_to_mask_enable
                               ? (rb_colorcontrol.value >> 24) | (1 << 8)
                               : 0;
  update_dirty_uint32_cmp(system_constants_.alpha_to_mask, alpha_to_mask);
  system_constants_.alpha_to_mask = alpha_to_mask;

  // Color exponent bias
  for (uint32_t i = 0; i < 4; ++i) {
    int32_t color_exp_bias = color_infos[i].color_exp_bias;
    // Fixed-point render targets (k_16_16 / k_16_16_16_16) are backed by
    // *_SNORM in the host render targets path. If full-range emulation is
    // requested, remap from -32...32 to -1...1 by dividing the output values
    // by 32.
    if (color_infos[i].color_format ==
        xenos::ColorRenderTargetFormat::k_16_16) {
      if (!render_target_cache_->IsFixedRG16TruncatedToMinus1To1()) {
        color_exp_bias -= 5;
      }
    } else if (color_infos[i].color_format ==
               xenos::ColorRenderTargetFormat::k_16_16_16_16) {
      if (!render_target_cache_->IsFixedRGBA16TruncatedToMinus1To1()) {
        color_exp_bias -= 5;
      }
    }
    auto color_exp_bias_scale = xe::memory::Reinterpret<float>(
        int32_t(0x3F800000 + (color_exp_bias << 23)));
    update_dirty_floatmask(system_constants_.color_exp_bias[i],
                           color_exp_bias_scale);
    system_constants_.color_exp_bias[i] = color_exp_bias_scale;
  }

  // Blend constants (used by EDRAM and for host blending)
  float blend_red = regs.Get<float>(XE_GPU_REG_RB_BLEND_RED);
  float blend_green = regs.Get<float>(XE_GPU_REG_RB_BLEND_GREEN);
  float blend_blue = regs.Get<float>(XE_GPU_REG_RB_BLEND_BLUE);
  float blend_alpha = regs.Get<float>(XE_GPU_REG_RB_BLEND_ALPHA);
  update_dirty_floatmask(system_constants_.edram_blend_constant[0], blend_red);
  update_dirty_floatmask(system_constants_.edram_blend_constant[1],
                         blend_green);
  update_dirty_floatmask(system_constants_.edram_blend_constant[2], blend_blue);
  update_dirty_floatmask(system_constants_.edram_blend_constant[3],
                         blend_alpha);
  system_constants_.edram_blend_constant[0] = blend_red;
  system_constants_.edram_blend_constant[1] = blend_green;
  system_constants_.edram_blend_constant[2] = blend_blue;
  system_constants_.edram_blend_constant[3] = blend_alpha;

  dirty |= ArchFloatMaskSignbit(dirty_float_mask);
  cbuffer_binding_system_up_to_date_ &= !dirty;
}

#define COMMAND_PROCESSOR MetalCommandProcessor
#include "../pm4_command_processor_implement.h"
#undef COMMAND_PROCESSOR

}  // namespace metal
}  // namespace gpu
}  // namespace xe
