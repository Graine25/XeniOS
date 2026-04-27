/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/metal/metal_pipeline_cache.h"

#include <dispatch/dispatch.h>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <utility>
#include <vector>

#include "third_party/metal-cpp/Foundation/NSProcessInfo.hpp"
#include "third_party/metal-cpp/Foundation/NSURL.hpp"

#include "third_party/fmt/include/fmt/format.h"
#include "xenia/base/assert.h"
#include "xenia/base/cvar.h"
#include "xenia/base/filesystem.h"
#include "xenia/base/logging.h"
#include "xenia/base/math.h"
#include "xenia/base/xxhash.h"
#include "xenia/gpu/draw_util.h"
#include "xenia/gpu/gpu_flags.h"
#include "xenia/gpu/metal/metal_shader_cache.h"
#include "xenia/gpu/registers.h"
#include "xenia/gpu/xenos.h"

#include "xenia/gpu/metal/metal_shader_converter.h"
using BYTE = uint8_t;
#include "xenia/gpu/metal/d3d12_5_1_bytecode/adaptive_quad_hs.h"
#include "xenia/gpu/metal/d3d12_5_1_bytecode/adaptive_triangle_hs.h"
#include "xenia/gpu/metal/d3d12_5_1_bytecode/continuous_quad_1cp_hs.h"
#include "xenia/gpu/metal/d3d12_5_1_bytecode/continuous_quad_4cp_hs.h"
#include "xenia/gpu/metal/d3d12_5_1_bytecode/continuous_triangle_1cp_hs.h"
#include "xenia/gpu/metal/d3d12_5_1_bytecode/continuous_triangle_3cp_hs.h"
#include "xenia/gpu/metal/d3d12_5_1_bytecode/discrete_quad_1cp_hs.h"
#include "xenia/gpu/metal/d3d12_5_1_bytecode/discrete_quad_4cp_hs.h"
#include "xenia/gpu/metal/d3d12_5_1_bytecode/discrete_triangle_1cp_hs.h"
#include "xenia/gpu/metal/d3d12_5_1_bytecode/discrete_triangle_3cp_hs.h"
#include "xenia/gpu/metal/d3d12_5_1_bytecode/tessellation_adaptive_vs.h"
#include "xenia/gpu/metal/d3d12_5_1_bytecode/tessellation_indexed_vs.h"
// Metal IR Converter Runtime - defines IRDescriptorTableEntry and bind points
#define IR_RUNTIME_METALCPP
#include "third_party/metal-shader-converter/include/metal_irconverter_runtime.h"

#ifndef DISPATCH_DATA_DESTRUCTOR_NONE
#define DISPATCH_DATA_DESTRUCTOR_NONE DISPATCH_DATA_DESTRUCTOR_DEFAULT
#endif

DECLARE_bool(async_shader_compilation);
DECLARE_bool(depth_float24_convert_in_pixel_shader);
DECLARE_bool(depth_float24_round);

DEFINE_int32(metal_pipeline_creation_threads, -1,
             "Number of threads for background pipeline compilation. "
             "-1 = auto (75% of cores), 0 = disabled (synchronous).",
             "Metal");

namespace xe {
namespace gpu {
namespace metal {

namespace {

void LogMetalErrorDetails(const char* label, NS::Error* error) {
  if (!error) {
    return;
  }
  const char* desc = error->localizedDescription()
                         ? error->localizedDescription()->utf8String()
                         : nullptr;
  const char* failure = error->localizedFailureReason()
                            ? error->localizedFailureReason()->utf8String()
                            : nullptr;
  const char* recovery =
      error->localizedRecoverySuggestion()
          ? error->localizedRecoverySuggestion()->utf8String()
          : nullptr;
  const char* domain =
      error->domain() ? error->domain()->utf8String() : nullptr;
  int64_t code = error->code();
  XELOGE("{}: domain={} code={} desc='{}' failure='{}' recovery='{}'", label,
         domain ? domain : "<null>", code, desc ? desc : "<null>",
         failure ? failure : "<null>", recovery ? recovery : "<null>");
  NS::Dictionary* user_info = error->userInfo();
  if (user_info) {
    auto* info_desc = user_info->description();
    XELOGE("{}: userInfo={}", label,
           info_desc ? info_desc->utf8String() : "<null>");
  }
}

void EnsureDepthFormatForDepthWritingFragment(const char* pipeline_name,
                                              bool fragment_writes_depth,
                                              MTL::PixelFormat* depth_format) {
  if (!fragment_writes_depth || !depth_format ||
      *depth_format != MTL::PixelFormatInvalid) {
    return;
  }
  *depth_format = MTL::PixelFormatDepth32Float;
  static bool logged = false;
  if (!logged) {
    logged = true;
    XELOGW(
        "{}: fragment writes depth without a bound depth attachment; "
        "using Depth32Float pipeline fallback",
        pipeline_name);
  }
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

MTL::ColorWriteMask ToMetalColorWriteMask(uint32_t write_mask) {
  MTL::ColorWriteMask mtl_mask = MTL::ColorWriteMaskNone;
  if (write_mask & 0x1) {
    mtl_mask |= MTL::ColorWriteMaskRed;
  }
  if (write_mask & 0x2) {
    mtl_mask |= MTL::ColorWriteMaskGreen;
  }
  if (write_mask & 0x4) {
    mtl_mask |= MTL::ColorWriteMaskBlue;
  }
  if (write_mask & 0x8) {
    mtl_mask |= MTL::ColorWriteMaskAlpha;
  }
  return mtl_mask;
}

MTL::BlendOperation ToMetalBlendOperation(xenos::BlendOp blend_op) {
  static const MTL::BlendOperation kBlendOpMap[8] = {
      MTL::BlendOperationAdd,              // 0
      MTL::BlendOperationSubtract,         // 1
      MTL::BlendOperationMin,              // 2
      MTL::BlendOperationMax,              // 3
      MTL::BlendOperationReverseSubtract,  // 4
      MTL::BlendOperationAdd,              // 5
      MTL::BlendOperationAdd,              // 6
      MTL::BlendOperationAdd,              // 7
  };
  return kBlendOpMap[uint32_t(blend_op) & 0x7];
}

MTL::BlendFactor ToMetalBlendFactorRgb(xenos::BlendFactor blend_factor) {
  static const MTL::BlendFactor kBlendFactorMap[32] = {
      /*  0 */ MTL::BlendFactorZero,
      /*  1 */ MTL::BlendFactorOne,
      /*  2 */ MTL::BlendFactorZero,  // ?
      /*  3 */ MTL::BlendFactorZero,  // ?
      /*  4 */ MTL::BlendFactorSourceColor,
      /*  5 */ MTL::BlendFactorOneMinusSourceColor,
      /*  6 */ MTL::BlendFactorSourceAlpha,
      /*  7 */ MTL::BlendFactorOneMinusSourceAlpha,
      /*  8 */ MTL::BlendFactorDestinationColor,
      /*  9 */ MTL::BlendFactorOneMinusDestinationColor,
      /* 10 */ MTL::BlendFactorDestinationAlpha,
      /* 11 */ MTL::BlendFactorOneMinusDestinationAlpha,
      /* 12 */ MTL::BlendFactorBlendColor,  // CONSTANT_COLOR
      /* 13 */ MTL::BlendFactorOneMinusBlendColor,
      /* 14 */ MTL::BlendFactorBlendAlpha,  // CONSTANT_ALPHA
      /* 15 */ MTL::BlendFactorOneMinusBlendAlpha,
      /* 16 */ MTL::BlendFactorSourceAlphaSaturated,
  };
  return kBlendFactorMap[uint32_t(blend_factor) & 0x1F];
}

MTL::BlendFactor ToMetalBlendFactorAlpha(xenos::BlendFactor blend_factor) {
  static const MTL::BlendFactor kBlendFactorAlphaMap[32] = {
      /*  0 */ MTL::BlendFactorZero,
      /*  1 */ MTL::BlendFactorOne,
      /*  2 */ MTL::BlendFactorZero,  // ?
      /*  3 */ MTL::BlendFactorZero,  // ?
      /*  4 */ MTL::BlendFactorSourceAlpha,
      /*  5 */ MTL::BlendFactorOneMinusSourceAlpha,
      /*  6 */ MTL::BlendFactorSourceAlpha,
      /*  7 */ MTL::BlendFactorOneMinusSourceAlpha,
      /*  8 */ MTL::BlendFactorDestinationAlpha,
      /*  9 */ MTL::BlendFactorOneMinusDestinationAlpha,
      /* 10 */ MTL::BlendFactorDestinationAlpha,
      /* 11 */ MTL::BlendFactorOneMinusDestinationAlpha,
      /* 12 */ MTL::BlendFactorBlendAlpha,
      /* 13 */ MTL::BlendFactorOneMinusBlendAlpha,
      /* 14 */ MTL::BlendFactorBlendAlpha,
      /* 15 */ MTL::BlendFactorOneMinusBlendAlpha,
      /* 16 */ MTL::BlendFactorSourceAlphaSaturated,
  };
  return kBlendFactorAlphaMap[uint32_t(blend_factor) & 0x1F];
}

void ApplyBlendStateToDescriptor(
    MTL::RenderPipelineColorAttachmentDescriptorArray* color_attachments,
    uint32_t normalized_color_mask, const uint32_t blendcontrol[4]) {
  for (uint32_t i = 0; i < 4; ++i) {
    auto* color_attachment = color_attachments->object(i);
    if (color_attachment->pixelFormat() == MTL::PixelFormatInvalid) {
      color_attachment->setWriteMask(MTL::ColorWriteMaskNone);
      color_attachment->setBlendingEnabled(false);
      continue;
    }

    uint32_t rt_write_mask = (normalized_color_mask >> (i * 4)) & 0xF;
    color_attachment->setWriteMask(ToMetalColorWriteMask(rt_write_mask));
    if (!rt_write_mask) {
      color_attachment->setBlendingEnabled(false);
      continue;
    }

    reg::RB_BLENDCONTROL bc = {};
    bc.value = blendcontrol[i];
    MTL::BlendFactor src_rgb = ToMetalBlendFactorRgb(bc.color_srcblend);
    MTL::BlendFactor dst_rgb = ToMetalBlendFactorRgb(bc.color_destblend);
    MTL::BlendOperation op_rgb = ToMetalBlendOperation(bc.color_comb_fcn);
    MTL::BlendFactor src_alpha = ToMetalBlendFactorAlpha(bc.alpha_srcblend);
    MTL::BlendFactor dst_alpha = ToMetalBlendFactorAlpha(bc.alpha_destblend);
    MTL::BlendOperation op_alpha = ToMetalBlendOperation(bc.alpha_comb_fcn);

    bool blending_enabled =
        src_rgb != MTL::BlendFactorOne || dst_rgb != MTL::BlendFactorZero ||
        op_rgb != MTL::BlendOperationAdd || src_alpha != MTL::BlendFactorOne ||
        dst_alpha != MTL::BlendFactorZero || op_alpha != MTL::BlendOperationAdd;
    color_attachment->setBlendingEnabled(blending_enabled);
    if (blending_enabled) {
      color_attachment->setSourceRGBBlendFactor(src_rgb);
      color_attachment->setDestinationRGBBlendFactor(dst_rgb);
      color_attachment->setRgbBlendOperation(op_rgb);
      color_attachment->setSourceAlphaBlendFactor(src_alpha);
      color_attachment->setDestinationAlphaBlendFactor(dst_alpha);
      color_attachment->setAlphaBlendOperation(op_alpha);
    }
  }
}

IRFormat MapVertexFormatToIRFormat(
    const ParsedVertexFetchInstruction::Attributes& attrs) {
  using xenos::VertexFormat;
  switch (attrs.data_format) {
    case VertexFormat::k_8_8_8_8:
      if (attrs.is_integer) {
        return attrs.is_signed ? IRFormatR8G8B8A8Sint : IRFormatR8G8B8A8Uint;
      }
      return attrs.is_signed ? IRFormatR8G8B8A8Snorm : IRFormatR8G8B8A8Unorm;
    case VertexFormat::k_2_10_10_10:
      if (attrs.is_integer) {
        return IRFormatR10G10B10A2Uint;
      }
      return IRFormatR10G10B10A2Unorm;
    case VertexFormat::k_10_11_11:
    case VertexFormat::k_11_11_10:
      return IRFormatR11G11B10Float;
    case VertexFormat::k_16_16:
      if (attrs.is_integer) {
        return attrs.is_signed ? IRFormatR16G16Sint : IRFormatR16G16Uint;
      }
      return attrs.is_signed ? IRFormatR16G16Snorm : IRFormatR16G16Unorm;
    case VertexFormat::k_16_16_16_16:
      if (attrs.is_integer) {
        return attrs.is_signed ? IRFormatR16G16B16A16Sint
                               : IRFormatR16G16B16A16Uint;
      }
      return attrs.is_signed ? IRFormatR16G16B16A16Snorm
                             : IRFormatR16G16B16A16Unorm;
    case VertexFormat::k_16_16_FLOAT:
      return IRFormatR16G16Float;
    case VertexFormat::k_16_16_16_16_FLOAT:
      return IRFormatR16G16B16A16Float;
    case VertexFormat::k_32:
      if (attrs.is_integer) {
        return attrs.is_signed ? IRFormatR32Sint : IRFormatR32Uint;
      }
      return IRFormatR32Float;
    case VertexFormat::k_32_32:
      if (attrs.is_integer) {
        return attrs.is_signed ? IRFormatR32G32Sint : IRFormatR32G32Uint;
      }
      return IRFormatR32G32Float;
    case VertexFormat::k_32_FLOAT:
      return IRFormatR32Float;
    case VertexFormat::k_32_32_FLOAT:
      return IRFormatR32G32Float;
    case VertexFormat::k_32_32_32_FLOAT:
      return IRFormatR32G32B32Float;
    case VertexFormat::k_32_32_32_32:
      if (attrs.is_integer) {
        return attrs.is_signed ? IRFormatR32G32B32A32Sint
                               : IRFormatR32G32B32A32Uint;
      }
      return IRFormatR32G32B32A32Float;
    case VertexFormat::k_32_32_32_32_FLOAT:
      return IRFormatR32G32B32A32Float;
    default:
      return IRFormatUnknown;
  }
}

constexpr uint32_t kPipelineDiskCacheMagic = 0x43504D58;  // 'XMPC'
constexpr uint32_t kPipelineDiskCacheVersion = 2;
constexpr size_t kPipelineDiskCacheMaxEntrySize = 1 << 20;

XEPACKEDSTRUCT(PipelineDiskCacheHeader, {
  uint32_t magic;
  uint32_t version;
  uint32_t reserved[2];
});

XEPACKEDSTRUCT(PipelineDiskCacheEntryHeader, {
  uint32_t entry_size;
  uint32_t reserved;
});

XEPACKEDSTRUCT(PipelineDiskCacheEntryBase, {
  uint64_t pipeline_key;
  uint64_t vertex_shader_cache_key;
  uint64_t pixel_shader_cache_key;
  uint32_t sample_count;
  uint32_t depth_format;
  uint32_t stencil_format;
  uint32_t color_formats[4];
  uint32_t normalized_color_mask;
  uint32_t alpha_to_mask_enable;
  uint32_t blendcontrol[4];
  uint32_t vertex_attribute_count;
  uint32_t vertex_layout_count;
});

static_assert(sizeof(PipelineDiskCacheHeader) == 16,
              "Unexpected pipeline disk cache header size.");
static_assert(sizeof(PipelineDiskCacheEntryHeader) == 8,
              "Unexpected pipeline disk cache entry header size.");

}  // namespace

// ---------------------------------------------------------------------------
// ResolvePipelineRenderingKey (shared fixed-function key derivation)
// ---------------------------------------------------------------------------

PipelineRenderingKey ResolvePipelineRenderingKey(
    const RegisterFile& regs,
    const MetalShader::MetalTranslation* pixel_translation,
    bool use_fallback_pixel_shader) {
  PipelineRenderingKey key = {};

  uint32_t pixel_shader_writes_color_targets =
      (use_fallback_pixel_shader || !pixel_translation)
          ? 0
          : pixel_translation->shader().writes_color_targets();
  key.normalized_color_mask = pixel_shader_writes_color_targets
                                  ? draw_util::GetNormalizedColorMask(
                                        regs, pixel_shader_writes_color_targets)
                                  : 0;
  auto rb_colorcontrol = regs.Get<reg::RB_COLORCONTROL>();
  key.alpha_to_mask_enable = rb_colorcontrol.alpha_to_mask_enable ? 1 : 0;
  for (uint32_t i = 0; i < 4; ++i) {
    key.blendcontrol[i] = regs.Get<reg::RB_BLENDCONTROL>(
                                  reg::RB_BLENDCONTROL::rt_register_indices[i])
                              .value;
  }
  return key;
}

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

MetalPipelineCache::MetalPipelineCache(MTL::Device* device,
                                       const RegisterFile& register_file)
    : device_(device), register_file_(register_file) {}

MetalPipelineCache::~MetalPipelineCache() {
  // Shut down async pipeline creation threads.
  {
    std::lock_guard<std::mutex> lock(creation_request_lock_);
    creation_threads_shutdown_ = true;
  }
  creation_request_cond_.notify_all();
  for (auto& thread : creation_threads_) {
    if (thread.joinable()) thread.join();
  }
  creation_threads_.clear();

  // Release MSC pipeline caches.
  for (auto& pair : pipeline_cache_) {
    if (pair.second) {
      auto* ps = pair.second->state.load(std::memory_order_relaxed);
      if (ps) {
        ps->release();
      }
    }
  }
  pipeline_cache_.clear();

  for (auto& pair : geometry_pipeline_cache_) {
    if (pair.second.pipeline) {
      pair.second.pipeline->release();
    }
  }
  geometry_pipeline_cache_.clear();

  for (auto& pair : geometry_vertex_stage_cache_) {
    if (pair.second.library) {
      pair.second.library->release();
    }
    if (pair.second.stage_in_library) {
      pair.second.stage_in_library->release();
    }
  }
  geometry_vertex_stage_cache_.clear();

  for (auto& pair : geometry_shader_stage_cache_) {
    if (pair.second.library) {
      pair.second.library->release();
    }
  }
  geometry_shader_stage_cache_.clear();

  if (depth_only_pixel_library_) {
    depth_only_pixel_library_->release();
    depth_only_pixel_library_ = nullptr;
  }
  depth_only_pixel_function_name_.clear();

  shader_cache_.clear();
  shader_translator_.reset();
  dxbc_to_dxil_converter_.reset();
  metal_shader_converter_.reset();

  ShutdownShaderStorage();
}

// ---------------------------------------------------------------------------
// Shader translation initialization
// ---------------------------------------------------------------------------

bool MetalPipelineCache::InitializeShaderTranslation(
    bool gamma_render_target_as_unorm8, bool msaa_2x_supported,
    uint32_t draw_resolution_scale_x, uint32_t draw_resolution_scale_y) {
  bool edram_rov_used = false;

  XELOGI(
      "DxbcShaderTranslator init: gamma_as_unorm8={}, msaa_2x={}, scale={}x{}",
      gamma_render_target_as_unorm8, msaa_2x_supported, draw_resolution_scale_x,
      draw_resolution_scale_y);

  shader_translator_ = std::make_unique<DxbcShaderTranslator>(
      ui::GraphicsProvider::GpuVendorID::kApple, true, edram_rov_used,
      gamma_render_target_as_unorm8, msaa_2x_supported, draw_resolution_scale_x,
      draw_resolution_scale_y,
      false);  // force_emit_source_map

  dxbc_to_dxil_converter_ = std::make_unique<DxbcToDxilConverter>();
  if (!dxbc_to_dxil_converter_->Initialize()) {
    XELOGE("Failed to initialize DXBC to DXIL converter");
    return false;
  }

  metal_shader_converter_ = std::make_unique<MetalShaderConverter>();
  if (!metal_shader_converter_->Initialize()) {
    XELOGE("Failed to initialize Metal Shader Converter");
    return false;
  }

  // Configure MSC minimum targets.
  if (device_) {
    IRGPUFamily min_family = IRGPUFamilyMetal3;
    if (device_->supportsFamily(MTL::GPUFamilyApple10)) {
      min_family = IRGPUFamilyApple10;
    } else if (device_->supportsFamily(MTL::GPUFamilyApple9)) {
      min_family = IRGPUFamilyApple9;
    } else if (device_->supportsFamily(MTL::GPUFamilyApple8)) {
      min_family = IRGPUFamilyApple8;
    } else if (device_->supportsFamily(MTL::GPUFamilyApple7)) {
      min_family = IRGPUFamilyApple7;
    } else if (device_->supportsFamily(MTL::GPUFamilyApple6)) {
      min_family = IRGPUFamilyApple6;
    } else if (device_->supportsFamily(MTL::GPUFamilyMac2) ||
               device_->supportsFamily(MTL::GPUFamilyMetal4) ||
               device_->supportsFamily(MTL::GPUFamilyMetal3)) {
      min_family = IRGPUFamilyMetal3;
    }

    NS::OperatingSystemVersion os_version =
        NS::ProcessInfo::processInfo()->operatingSystemVersion();
    std::ostringstream version_stream;
    version_stream << os_version.majorVersion << "." << os_version.minorVersion
                   << "." << os_version.patchVersion;
    metal_shader_converter_->SetMinimumTarget(
        min_family, IROperatingSystem_macOS, version_stream.str());
  }

  // Spawn async pipeline creation threads if enabled.
  int32_t thread_count_config = cvars::metal_pipeline_creation_threads;
  uint32_t logical_cores = std::thread::hardware_concurrency();
  uint32_t thread_count = 0;
  if (thread_count_config < 0) {
    thread_count = std::max(1u, logical_cores * 3 / 4);
  } else {
    thread_count =
        std::min(static_cast<uint32_t>(thread_count_config), logical_cores);
  }
  if (thread_count > 0 && cvars::async_shader_compilation) {
    creation_threads_.reserve(thread_count);
    for (uint32_t i = 0; i < thread_count; ++i) {
      creation_threads_.emplace_back(&MetalPipelineCache::CreationThread, this,
                                     i);
    }
    XELOGI("MetalPipelineCache: spawned {} async pipeline creation threads",
           thread_count);
  }

  return true;
}

// ---------------------------------------------------------------------------
// Shader storage lifecycle
// ---------------------------------------------------------------------------

void MetalPipelineCache::InitializeShaderStorage(
    const std::filesystem::path& cache_root, uint32_t title_id, bool blocking) {
  InitializeShaderStorageInternal(cache_root, title_id, blocking);
}

bool MetalPipelineCache::InitializeShaderStorageInternal(
    const std::filesystem::path& cache_root, uint32_t title_id, bool blocking) {
  ShutdownShaderStorage();

  if (!device_) {
    XELOGW("Metal shader storage init skipped (no device)");
    return false;
  }

  shader_storage_root_ = cache_root / "shaders" / "metal";
  shader_storage_local_root_ =
      shader_storage_root_ / "local" / GetShaderStorageDeviceTag();
  shader_storage_title_root_ = shader_storage_local_root_ /
                               GetShaderStorageAbiTag() /
                               fmt::format("{:08X}", title_id);

  std::error_code ec;
  std::filesystem::create_directories(shader_storage_title_root_, ec);
  if (ec) {
    XELOGW("Metal shader storage: Failed to create {}: {}",
           shader_storage_title_root_.string(), ec.message());
    return false;
  }

  metallib_cache_dir_ = shader_storage_title_root_ / "metallib";
  if (::cvars::metal_shader_disk_cache && g_metal_shader_cache) {
    g_metal_shader_cache->Initialize(metallib_cache_dir_);
  }

  const char* path_suffix = "rtv-bindless";
  std::string storage_suffix =
      fmt::format("{}.{}", path_suffix, GetShaderStorageAbiTag());

  pipeline_disk_cache_path_ =
      shader_storage_title_root_ /
      fmt::format("{:08X}.{}.metal.pipelines", title_id, storage_suffix);
  pipeline_binary_archive_path_ =
      shader_storage_title_root_ /
      fmt::format("{:08X}.{}.metal.binarchive", title_id, storage_suffix);

  if (::cvars::metal_pipeline_disk_cache) {
    LoadPipelineDiskCache(pipeline_disk_cache_path_,
                          &pipeline_disk_cache_entries_);
  }

  if (::cvars::metal_pipeline_binary_archive) {
    InitializePipelineBinaryArchive(pipeline_binary_archive_path_);
  }

  bool prewarm_binary_archive = false;
  {
    std::lock_guard<std::mutex> lock(pipeline_binary_archive_mutex_);
    prewarm_binary_archive = pipeline_binary_archive_ != nullptr;
  }
  if (blocking && prewarm_binary_archive &&
      !pipeline_disk_cache_entries_.empty()) {
    PrewarmPipelineBinaryArchive(pipeline_disk_cache_entries_);
  }

  return true;
}

void MetalPipelineCache::ShutdownShaderStorage() {
  {
    std::lock_guard<std::mutex> lock(pipeline_binary_archive_mutex_);
    if (pipeline_binary_archive_) {
      if (pipeline_binary_archive_dirty_) {
        NS::String* path_string =
            NS::String::string(pipeline_binary_archive_path_.string().c_str(),
                               NS::UTF8StringEncoding);
        NS::URL* url = NS::URL::fileURLWithPath(path_string);
        NS::Error* error = nullptr;
        if (!pipeline_binary_archive_->serializeToURL(url, &error)) {
          if (error) {
            XELOGW("Metal binary archive serialize failed: {}",
                   error->localizedDescription()->utf8String());
          }
        }
        pipeline_binary_archive_dirty_ = false;
      }
      pipeline_binary_archive_->release();
      pipeline_binary_archive_ = nullptr;
    }
  }
  if (pipeline_disk_cache_file_) {
    std::fclose(pipeline_disk_cache_file_);
    pipeline_disk_cache_file_ = nullptr;
  }
  pipeline_disk_cache_keys_.clear();
  pipeline_disk_cache_entries_.clear();

  if (g_metal_shader_cache) {
    g_metal_shader_cache->Shutdown();
  }
}

std::string MetalPipelineCache::GetShaderStorageDeviceTag() const {
  std::string tag = "unknown";
  if (device_ && device_->name()) {
    tag = device_->name()->utf8String();
  }

  for (char& ch : tag) {
    if (!std::isalnum(static_cast<unsigned char>(ch))) {
      ch = '_';
    }
  }
  return tag;
}

std::string MetalPipelineCache::GetShaderStorageAbiTag() const {
  return fmt::format("msc{:08X}-ps{:08X}-sh{:08X}",
                     DxbcShaderTranslator::Modification::kVersion,
                     kPipelineDiskCacheVersion,
                     MetalShaderCache::kStorageVersion);
}

// ---------------------------------------------------------------------------
// Pipeline disk cache
// ---------------------------------------------------------------------------

bool MetalPipelineCache::LoadPipelineDiskCache(
    const std::filesystem::path& path,
    std::vector<PipelineDiskCacheEntry>* entries) {
  if (!entries) {
    return false;
  }
  entries->clear();
  pipeline_disk_cache_keys_.clear();

  pipeline_disk_cache_file_ = xe::filesystem::OpenFile(path, "a+b");
  if (!pipeline_disk_cache_file_) {
    XELOGW("Metal pipeline disk cache: Failed to open {}", path.string());
    return false;
  }

  PipelineDiskCacheHeader header = {};
  if (std::fread(&header, sizeof(header), 1, pipeline_disk_cache_file_) != 1 ||
      header.magic != kPipelineDiskCacheMagic ||
      header.version != kPipelineDiskCacheVersion) {
    header.magic = kPipelineDiskCacheMagic;
    header.version = kPipelineDiskCacheVersion;
    header.reserved[0] = 0;
    header.reserved[1] = 0;
    xe::filesystem::Seek(pipeline_disk_cache_file_, 0, SEEK_SET);
    std::fwrite(&header, sizeof(header), 1, pipeline_disk_cache_file_);
    std::fflush(pipeline_disk_cache_file_);
    xe::filesystem::Seek(pipeline_disk_cache_file_, 0, SEEK_END);
    return true;
  }

  while (true) {
    PipelineDiskCacheEntryHeader entry_header = {};
    if (std::fread(&entry_header, sizeof(entry_header), 1,
                   pipeline_disk_cache_file_) != 1) {
      break;
    }

    if (entry_header.entry_size < sizeof(PipelineDiskCacheEntryBase) ||
        entry_header.entry_size > kPipelineDiskCacheMaxEntrySize) {
      break;
    }

    PipelineDiskCacheEntryBase base = {};
    if (std::fread(&base, sizeof(base), 1, pipeline_disk_cache_file_) != 1) {
      break;
    }

    size_t expected_size = sizeof(PipelineDiskCacheEntryBase) +
                           size_t(base.vertex_attribute_count) *
                               sizeof(PipelineDiskCacheVertexAttribute) +
                           size_t(base.vertex_layout_count) *
                               sizeof(PipelineDiskCacheVertexLayout);
    if (entry_header.entry_size != expected_size) {
      xe::filesystem::Seek(
          pipeline_disk_cache_file_,
          entry_header.entry_size - sizeof(PipelineDiskCacheEntryBase),
          SEEK_CUR);
      continue;
    }

    PipelineDiskCacheEntry entry = {};
    entry.pipeline_key = base.pipeline_key;
    entry.vertex_shader_cache_key = base.vertex_shader_cache_key;
    entry.pixel_shader_cache_key = base.pixel_shader_cache_key;
    entry.sample_count = base.sample_count;
    entry.depth_format = base.depth_format;
    entry.stencil_format = base.stencil_format;
    std::memcpy(entry.color_formats, base.color_formats,
                sizeof(base.color_formats));
    entry.normalized_color_mask = base.normalized_color_mask;
    entry.alpha_to_mask_enable = base.alpha_to_mask_enable;
    std::memcpy(entry.blendcontrol, base.blendcontrol,
                sizeof(base.blendcontrol));

    entry.vertex_attributes.resize(base.vertex_attribute_count);
    if (base.vertex_attribute_count) {
      if (std::fread(entry.vertex_attributes.data(),
                     sizeof(PipelineDiskCacheVertexAttribute),
                     base.vertex_attribute_count, pipeline_disk_cache_file_) !=
          base.vertex_attribute_count) {
        break;
      }
    }
    entry.vertex_layouts.resize(base.vertex_layout_count);
    if (base.vertex_layout_count) {
      if (std::fread(entry.vertex_layouts.data(),
                     sizeof(PipelineDiskCacheVertexLayout),
                     base.vertex_layout_count,
                     pipeline_disk_cache_file_) != base.vertex_layout_count) {
        break;
      }
    }

    entries->push_back(std::move(entry));
    pipeline_disk_cache_keys_.insert(base.pipeline_key);
  }

  xe::filesystem::Seek(pipeline_disk_cache_file_, 0, SEEK_END);
  return true;
}

bool MetalPipelineCache::AppendPipelineDiskCacheEntry(
    const PipelineDiskCacheEntry& entry) {
  if (!pipeline_disk_cache_file_) {
    return false;
  }
  if (!pipeline_disk_cache_keys_.insert(entry.pipeline_key).second) {
    return false;
  }

  PipelineDiskCacheEntryBase base = {};
  base.pipeline_key = entry.pipeline_key;
  base.vertex_shader_cache_key = entry.vertex_shader_cache_key;
  base.pixel_shader_cache_key = entry.pixel_shader_cache_key;
  base.sample_count = entry.sample_count;
  base.depth_format = entry.depth_format;
  base.stencil_format = entry.stencil_format;
  std::memcpy(base.color_formats, entry.color_formats,
              sizeof(base.color_formats));
  base.normalized_color_mask = entry.normalized_color_mask;
  base.alpha_to_mask_enable = entry.alpha_to_mask_enable;
  std::memcpy(base.blendcontrol, entry.blendcontrol, sizeof(base.blendcontrol));
  base.vertex_attribute_count =
      static_cast<uint32_t>(entry.vertex_attributes.size());
  base.vertex_layout_count = static_cast<uint32_t>(entry.vertex_layouts.size());

  size_t entry_size =
      sizeof(PipelineDiskCacheEntryBase) +
      entry.vertex_attributes.size() *
          sizeof(PipelineDiskCacheVertexAttribute) +
      entry.vertex_layouts.size() * sizeof(PipelineDiskCacheVertexLayout);
  if (entry_size > kPipelineDiskCacheMaxEntrySize) {
    return false;
  }

  PipelineDiskCacheEntryHeader entry_header = {};
  entry_header.entry_size = static_cast<uint32_t>(entry_size);

  std::fwrite(&entry_header, sizeof(entry_header), 1,
              pipeline_disk_cache_file_);
  std::fwrite(&base, sizeof(base), 1, pipeline_disk_cache_file_);
  if (!entry.vertex_attributes.empty()) {
    std::fwrite(entry.vertex_attributes.data(),
                sizeof(PipelineDiskCacheVertexAttribute),
                entry.vertex_attributes.size(), pipeline_disk_cache_file_);
  }
  if (!entry.vertex_layouts.empty()) {
    std::fwrite(entry.vertex_layouts.data(),
                sizeof(PipelineDiskCacheVertexLayout),
                entry.vertex_layouts.size(), pipeline_disk_cache_file_);
  }
  std::fflush(pipeline_disk_cache_file_);
  pipeline_disk_cache_entries_.push_back(entry);
  return true;
}

// ---------------------------------------------------------------------------
// Pipeline binary archive
// ---------------------------------------------------------------------------

bool MetalPipelineCache::InitializePipelineBinaryArchive(
    const std::filesystem::path& archive_path) {
  if (!device_) {
    return false;
  }
  std::lock_guard<std::mutex> lock(pipeline_binary_archive_mutex_);
  if (pipeline_binary_archive_) {
    pipeline_binary_archive_->release();
    pipeline_binary_archive_ = nullptr;
  }

  MTL::BinaryArchiveDescriptor* desc =
      MTL::BinaryArchiveDescriptor::alloc()->init();
  std::error_code ec;
  bool archive_exists = std::filesystem::exists(archive_path, ec);
  if (ec) {
    XELOGW("Metal binary archive existence check failed for {}: {}",
           archive_path.string(), ec.message());
    archive_exists = false;
  }
  if (archive_exists) {
    NS::String* path_string = NS::String::string(archive_path.string().c_str(),
                                                 NS::UTF8StringEncoding);
    NS::URL* url = NS::URL::fileURLWithPath(path_string);
    desc->setUrl(url);
  }

  NS::Error* error = nullptr;
  pipeline_binary_archive_ = device_->newBinaryArchive(desc, &error);
  if (!pipeline_binary_archive_ && archive_exists) {
    if (error) {
      XELOGW(
          "Metal binary archive load failed for existing file {}; retrying "
          "with a fresh archive: {}",
          archive_path.string(), error->localizedDescription()->utf8String());
    }
    desc->setUrl(nullptr);
    error = nullptr;
    pipeline_binary_archive_ = device_->newBinaryArchive(desc, &error);
  }
  desc->release();
  if (!pipeline_binary_archive_) {
    if (error) {
      XELOGW("Metal binary archive init failed: {}",
             error->localizedDescription()->utf8String());
    }
    return false;
  }
  pipeline_binary_archive_path_ = archive_path;
  pipeline_binary_archive_dirty_ = false;
  return true;
}

void MetalPipelineCache::SerializePipelineBinaryArchive() {
  std::lock_guard<std::mutex> lock(pipeline_binary_archive_mutex_);
  if (!pipeline_binary_archive_ || !pipeline_binary_archive_dirty_) {
    return;
  }
  NS::String* path_string = NS::String::string(
      pipeline_binary_archive_path_.string().c_str(), NS::UTF8StringEncoding);
  NS::URL* url = NS::URL::fileURLWithPath(path_string);
  NS::Error* error = nullptr;
  if (!pipeline_binary_archive_->serializeToURL(url, &error)) {
    if (error) {
      XELOGW("Metal binary archive serialize failed: {}",
             error->localizedDescription()->utf8String());
    }
  }
  pipeline_binary_archive_dirty_ = false;
}

void MetalPipelineCache::PrewarmPipelineBinaryArchive(
    const std::vector<PipelineDiskCacheEntry>& entries) {
  if (entries.empty()) {
    return;
  }
  if (!g_metal_shader_cache || !g_metal_shader_cache->IsInitialized()) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(pipeline_binary_archive_mutex_);
    if (!pipeline_binary_archive_) {
      return;
    }
  }

  size_t prewarmed = 0;
  for (const auto& entry : entries) {
    MetalShaderCache::CachedMetallib vs_cached;
    if (!g_metal_shader_cache->Load(entry.vertex_shader_cache_key,
                                    &vs_cached)) {
      continue;
    }

    NS::Error* error = nullptr;
    dispatch_data_t vs_data = dispatch_data_create(
        vs_cached.metallib_data.data(), vs_cached.metallib_data.size(), nullptr,
        DISPATCH_DATA_DESTRUCTOR_NONE);
    MTL::Library* vs_library = device_->newLibrary(vs_data, &error);
    dispatch_release(vs_data);
    if (!vs_library) {
      continue;
    }
    NS::String* vs_name = NS::String::string(vs_cached.function_name.c_str(),
                                             NS::UTF8StringEncoding);
    MTL::Function* vs_function = vs_library->newFunction(vs_name);
    if (!vs_function) {
      vs_library->release();
      continue;
    }

    MTL::Library* ps_library = nullptr;
    MTL::Function* ps_function = nullptr;
    if (entry.pixel_shader_cache_key) {
      MetalShaderCache::CachedMetallib ps_cached;
      if (g_metal_shader_cache->Load(entry.pixel_shader_cache_key,
                                     &ps_cached)) {
        dispatch_data_t ps_data = dispatch_data_create(
            ps_cached.metallib_data.data(), ps_cached.metallib_data.size(),
            nullptr, DISPATCH_DATA_DESTRUCTOR_NONE);
        ps_library = device_->newLibrary(ps_data, &error);
        dispatch_release(ps_data);
        if (ps_library) {
          NS::String* ps_name = NS::String::string(
              ps_cached.function_name.c_str(), NS::UTF8StringEncoding);
          ps_function = ps_library->newFunction(ps_name);
        }
      }
    }

    MTL::RenderPipelineDescriptor* desc =
        MTL::RenderPipelineDescriptor::alloc()->init();
    desc->setVertexFunction(vs_function);
    if (ps_function) {
      desc->setFragmentFunction(ps_function);
    }

    for (uint32_t i = 0; i < 4; ++i) {
      desc->colorAttachments()->object(i)->setPixelFormat(
          static_cast<MTL::PixelFormat>(entry.color_formats[i]));
    }
    desc->setDepthAttachmentPixelFormat(
        static_cast<MTL::PixelFormat>(entry.depth_format));
    desc->setStencilAttachmentPixelFormat(
        static_cast<MTL::PixelFormat>(entry.stencil_format));
    desc->setSampleCount(entry.sample_count);
    desc->setAlphaToCoverageEnabled(entry.alpha_to_mask_enable != 0);

    ApplyBlendStateToDescriptor(desc->colorAttachments(),
                                entry.normalized_color_mask,
                                entry.blendcontrol);

    if (!entry.vertex_attributes.empty() || !entry.vertex_layouts.empty()) {
      MTL::VertexDescriptor* vertex_desc =
          MTL::VertexDescriptor::alloc()->init();
      for (const auto& attr : entry.vertex_attributes) {
        auto* attr_desc =
            vertex_desc->attributes()->object(attr.attribute_index);
        attr_desc->setFormat(static_cast<MTL::VertexFormat>(attr.format));
        attr_desc->setOffset(attr.offset);
        attr_desc->setBufferIndex(attr.buffer_index);
      }
      for (const auto& layout : entry.vertex_layouts) {
        auto* layout_desc = vertex_desc->layouts()->object(layout.buffer_index);
        layout_desc->setStride(layout.stride);
        layout_desc->setStepFunction(
            static_cast<MTL::VertexStepFunction>(layout.step_function));
        layout_desc->setStepRate(layout.step_rate);
      }
      desc->setVertexDescriptor(vertex_desc);
      vertex_desc->release();
    }

    {
      std::lock_guard<std::mutex> lock(pipeline_binary_archive_mutex_);
      if (pipeline_binary_archive_) {
        NS::Array* archives = NS::Array::array(pipeline_binary_archive_);
        desc->setBinaryArchives(archives);
        if (pipeline_binary_archive_->addRenderPipelineFunctions(desc,
                                                                 &error)) {
          pipeline_binary_archive_dirty_ = true;
          ++prewarmed;
        }
      }
    }
    desc->release();
    vs_function->release();
    vs_library->release();
    if (ps_function) {
      ps_function->release();
    }
    if (ps_library) {
      ps_library->release();
    }
  }
}

// ---------------------------------------------------------------------------
// LoadShader
// ---------------------------------------------------------------------------

Shader* MetalPipelineCache::LoadShader(xenos::ShaderType shader_type,
                                       uint32_t guest_address,
                                       const uint32_t* host_address,
                                       uint32_t dword_count) {
  uint64_t hash = XXH3_64bits(host_address, dword_count * sizeof(uint32_t));

  auto it = shader_cache_.find(hash);
  if (it != shader_cache_.end()) {
    return it->second.get();
  }

  auto shader = std::make_unique<MetalShader>(shader_type, hash, host_address,
                                              dword_count);

  MetalShader* result = shader.get();
  shader_cache_[hash] = std::move(shader);

  XELOGD("Loaded {} shader at {:08X} ({} dwords, hash {:016X})",
         shader_type == xenos::ShaderType::kVertex ? "vertex" : "pixel",
         guest_address, dword_count, hash);

  return result;
}

// ---------------------------------------------------------------------------
// EnsureDepthOnlyPixelShader
// ---------------------------------------------------------------------------

bool MetalPipelineCache::EnsureDepthOnlyPixelShader() {
  if (depth_only_pixel_library_) {
    return true;
  }
  if (!shader_translator_ || !dxbc_to_dxil_converter_ ||
      !metal_shader_converter_) {
    XELOGE("Depth-only PS: shader translation not initialized");
    return false;
  }

  std::vector<uint8_t> dxbc_data =
      shader_translator_->CreateDepthOnlyPixelShader();
  if (dxbc_data.empty()) {
    XELOGE("Depth-only PS: failed to create DXBC");
    return false;
  }

  std::vector<uint8_t> dxil_data;
  std::string dxil_error;
  if (!dxbc_to_dxil_converter_->Convert(dxbc_data, dxil_data, &dxil_error)) {
    XELOGE("Depth-only PS: DXBC to DXIL conversion failed: {}", dxil_error);
    return false;
  }

  MetalShaderConversionResult result;
  if (!metal_shader_converter_->ConvertWithStage(MetalShaderStage::kFragment,
                                                 dxil_data, result)) {
    XELOGE("Depth-only PS: DXIL to Metal conversion failed: {}",
           result.error_message);
    return false;
  }

  NS::Error* error = nullptr;
  dispatch_data_t lib_data = dispatch_data_create(
      result.metallib_data.data(), result.metallib_data.size(), nullptr,
      DISPATCH_DATA_DESTRUCTOR_DEFAULT);
  depth_only_pixel_library_ = device_->newLibrary(lib_data, &error);
  dispatch_release(lib_data);
  if (!depth_only_pixel_library_) {
    XELOGE("Depth-only PS: Failed to create Metal library: {}",
           error ? error->localizedDescription()->utf8String() : "unknown");
    return false;
  }
  depth_only_pixel_function_name_ = result.function_name;
  if (depth_only_pixel_function_name_.empty()) {
    XELOGE("Depth-only PS: missing function name");
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Shader modification selection
// ---------------------------------------------------------------------------

DxbcShaderTranslator::Modification
MetalPipelineCache::GetCurrentVertexShaderModification(
    const Shader& shader, Shader::HostVertexShaderType host_vertex_shader_type,
    uint32_t interpolator_mask) const {
  const auto& regs = register_file_;

  DxbcShaderTranslator::Modification modification(
      shader_translator_->GetDefaultVertexShaderModification(
          shader.GetDynamicAddressableRegisterCount(
              regs.Get<reg::SQ_PROGRAM_CNTL>().vs_num_reg),
          host_vertex_shader_type));

  modification.vertex.interpolator_mask = interpolator_mask;

  auto pa_cl_clip_cntl = regs.Get<reg::PA_CL_CLIP_CNTL>();
  uint32_t user_clip_planes =
      pa_cl_clip_cntl.clip_disable ? 0 : pa_cl_clip_cntl.ucp_ena;
  modification.vertex.user_clip_plane_count = xe::bit_count(user_clip_planes);
  modification.vertex.user_clip_plane_cull =
      uint32_t(user_clip_planes && pa_cl_clip_cntl.ucp_cull_only_ena);
  modification.vertex.vertex_kill_and =
      uint32_t((shader.writes_point_size_edge_flag_kill_vertex() & 0b100) &&
               !pa_cl_clip_cntl.vtx_kill_or);

  modification.vertex.output_point_size =
      uint32_t((shader.writes_point_size_edge_flag_kill_vertex() & 0b001) &&
               regs.Get<reg::VGT_DRAW_INITIATOR>().prim_type ==
                   xenos::PrimitiveType::kPointList);

  return modification;
}

DxbcShaderTranslator::Modification
MetalPipelineCache::GetCurrentPixelShaderModification(
    const Shader& shader, uint32_t interpolator_mask, uint32_t param_gen_pos,
    reg::RB_DEPTHCONTROL normalized_depth_control) const {
  const auto& regs = register_file_;

  DxbcShaderTranslator::Modification modification(
      shader_translator_->GetDefaultPixelShaderModification(
          shader.GetDynamicAddressableRegisterCount(
              regs.Get<reg::SQ_PROGRAM_CNTL>().ps_num_reg)));

  modification.pixel.interpolator_mask = interpolator_mask;
  modification.pixel.interpolators_centroid =
      interpolator_mask &
      ~xenos::GetInterpolatorSamplingPattern(
          regs.Get<reg::RB_SURFACE_INFO>().msaa_samples,
          regs.Get<reg::SQ_CONTEXT_MISC>().sc_sample_cntl,
          regs.Get<reg::SQ_INTERPOLATOR_CNTL>().sampling_pattern);

  if (param_gen_pos < xenos::kMaxInterpolators) {
    modification.pixel.param_gen_enable = 1;
    modification.pixel.param_gen_interpolator = param_gen_pos;
    modification.pixel.param_gen_point =
        uint32_t(regs.Get<reg::VGT_DRAW_INITIATOR>().prim_type ==
                 xenos::PrimitiveType::kPointList);
  } else {
    modification.pixel.param_gen_enable = 0;
    modification.pixel.param_gen_interpolator = 0;
    modification.pixel.param_gen_point = 0;
  }

  using DepthStencilMode = DxbcShaderTranslator::Modification::DepthStencilMode;
  if (cvars::depth_float24_convert_in_pixel_shader &&
      normalized_depth_control.z_enable &&
      regs.Get<reg::RB_DEPTH_INFO>().depth_format ==
          xenos::DepthRenderTargetFormat::kD24FS8) {
    modification.pixel.depth_stencil_mode =
        cvars::depth_float24_round ? DepthStencilMode::kFloat24Rounding
                                   : DepthStencilMode::kFloat24Truncating;
  } else if (shader.implicit_early_z_write_allowed() &&
             (!shader.writes_color_target(0) ||
              !draw_util::DoesCoverageDependOnAlpha(
                  regs.Get<reg::RB_COLORCONTROL>()))) {
    modification.pixel.depth_stencil_mode = DepthStencilMode::kEarlyHint;
  } else {
    modification.pixel.depth_stencil_mode = DepthStencilMode::kNoModifiers;
  }

  // Check if MIN/MAX blend is used with non-trivial source factors.
  // Fixed-function blend ignores factors for MIN/MAX, but Xbox 360 applies
  // them. If the destination factor is ONE, we can pre-multiply the shader
  // output by the source factor to emulate this. Only RT0 is supported.
  modification.pixel.rt0_blend_rgb_factor_for_premult =
      xenos::BlendFactor::kOne;
  modification.pixel.rt0_blend_a_factor_for_premult = xenos::BlendFactor::kOne;

  if (shader.writes_color_target(0)) {
    auto blend_control = regs.Get<reg::RB_BLENDCONTROL>(
        reg::RB_BLENDCONTROL::rt_register_indices[0]);

    // Pre-multiply by kSrcAlpha for MIN/MAX blend ops when dstFactor is ONE.
    if ((blend_control.color_comb_fcn == xenos::BlendOp::kMin ||
         blend_control.color_comb_fcn == xenos::BlendOp::kMax) &&
        blend_control.color_srcblend == xenos::BlendFactor::kSrcAlpha &&
        blend_control.color_destblend == xenos::BlendFactor::kOne) {
      modification.pixel.rt0_blend_rgb_factor_for_premult =
          xenos::BlendFactor::kSrcAlpha;
    }

    if ((blend_control.alpha_comb_fcn == xenos::BlendOp::kMin ||
         blend_control.alpha_comb_fcn == xenos::BlendOp::kMax) &&
        blend_control.alpha_srcblend == xenos::BlendFactor::kSrcAlpha &&
        blend_control.alpha_destblend == xenos::BlendFactor::kOne) {
      modification.pixel.rt0_blend_a_factor_for_premult =
          xenos::BlendFactor::kSrcAlpha;
    }
  }

  return modification;
}

// ---------------------------------------------------------------------------
// GetOrCreatePipelineState (standard render pipeline)
// ---------------------------------------------------------------------------

MetalPipelineCache::PipelineHandle*
MetalPipelineCache::GetOrCreatePipelineState(
    MetalShader::MetalTranslation* vertex_translation,
    MetalShader::MetalTranslation* pixel_translation,
    const PipelineAttachmentFormats& attachment_formats,
    const PipelineRenderingKey& rendering_key) {
  if (!vertex_translation || !vertex_translation->metal_function()) {
    XELOGE("No valid vertex shader function");
    return nullptr;
  }

  struct PipelineKey {
    const void* vs;
    const void* ps;
    uint32_t sample_count;
    uint32_t depth_format;
    uint32_t stencil_format;
    uint32_t color_formats[4];
    uint32_t normalized_color_mask;
    uint32_t alpha_to_mask_enable;
    uint32_t blendcontrol[4];
  } key_data = {};
  key_data.vs = vertex_translation;
  key_data.ps = pixel_translation;
  key_data.sample_count = attachment_formats.sample_count;
  key_data.depth_format = uint32_t(attachment_formats.depth_format);
  key_data.stencil_format = uint32_t(attachment_formats.stencil_format);
  for (uint32_t i = 0; i < 4; ++i) {
    key_data.color_formats[i] = uint32_t(attachment_formats.color_formats[i]);
  }
  key_data.normalized_color_mask = rendering_key.normalized_color_mask;
  key_data.alpha_to_mask_enable = rendering_key.alpha_to_mask_enable;
  std::memcpy(key_data.blendcontrol, rendering_key.blendcontrol,
              sizeof(key_data.blendcontrol));
  uint64_t key = XXH3_64bits(&key_data, sizeof(key_data));

  // Check cache.
  auto it = pipeline_cache_.find(key);
  if (it != pipeline_cache_.end()) {
    return it->second.get();
  }

  // Prepare disk cache entry before creating the handle.
  PipelineDiskCacheEntry disk_entry;
  bool record_disk_entry =
      ::cvars::metal_pipeline_disk_cache && pipeline_disk_cache_file_;
  if (record_disk_entry) {
    disk_entry.pipeline_key = key;
    disk_entry.vertex_shader_cache_key = MetalShaderCache::GetCacheKey(
        vertex_translation->shader().ucode_data_hash(),
        vertex_translation->modification(),
        static_cast<uint32_t>(vertex_translation->shader().type()));
    if (pixel_translation) {
      disk_entry.pixel_shader_cache_key = MetalShaderCache::GetCacheKey(
          pixel_translation->shader().ucode_data_hash(),
          pixel_translation->modification(),
          static_cast<uint32_t>(pixel_translation->shader().type()));
    }
    disk_entry.sample_count = key_data.sample_count;
    disk_entry.depth_format = key_data.depth_format;
    disk_entry.stencil_format = key_data.stencil_format;
    std::memcpy(disk_entry.color_formats, key_data.color_formats,
                sizeof(key_data.color_formats));
    disk_entry.normalized_color_mask = key_data.normalized_color_mask;
    disk_entry.alpha_to_mask_enable = key_data.alpha_to_mask_enable;
    std::memcpy(disk_entry.blendcontrol, key_data.blendcontrol,
                sizeof(key_data.blendcontrol));
  }

  // Create a new handle.
  auto handle = std::make_unique<PipelineHandle>();
  handle->pending_vertex_translation = vertex_translation;
  handle->pending_pixel_translation = pixel_translation;
  handle->pending_formats = attachment_formats;
  handle->pending_normalized_color_mask = key_data.normalized_color_mask;
  std::memcpy(handle->pending_blendcontrol, key_data.blendcontrol,
              sizeof(key_data.blendcontrol));
  handle->pending_alpha_to_mask = (key_data.alpha_to_mask_enable != 0);

  PipelineHandle* raw_handle = handle.get();
  pipeline_cache_.emplace(key, std::move(handle));

  // Async path: enqueue for background compilation.
  if (cvars::async_shader_compilation && !creation_threads_.empty()) {
    {
      std::lock_guard<std::mutex> lock(creation_request_lock_);
      creation_queue_.push(PipelineCreationRequest{raw_handle});
    }
    creation_request_cond_.notify_one();
    return raw_handle;
  }

  // Synchronous path: create immediately.
  MTL::RenderPipelineState* pipeline = CreatePipelineFromHandle(raw_handle);
  raw_handle->state.store(pipeline, std::memory_order_release);
  raw_handle->pending_vertex_translation = nullptr;
  raw_handle->pending_pixel_translation = nullptr;

  if (!pipeline) {
    return nullptr;
  }

  if (record_disk_entry) {
    AppendPipelineDiskCacheEntry(disk_entry);
  }

  return raw_handle;
}

// ---------------------------------------------------------------------------
// CreatePipelineFromHandle (builds a MTL::RenderPipelineState from a handle)
// ---------------------------------------------------------------------------

MTL::RenderPipelineState* MetalPipelineCache::CreatePipelineFromHandle(
    const PipelineHandle* handle) {
  MetalShader::MetalTranslation* vertex_translation =
      handle->pending_vertex_translation;
  MetalShader::MetalTranslation* pixel_translation =
      handle->pending_pixel_translation;
  const PipelineAttachmentFormats& formats = handle->pending_formats;

  // Create pipeline descriptor.
  MTL::RenderPipelineDescriptor* desc =
      MTL::RenderPipelineDescriptor::alloc()->init();

  desc->setVertexFunction(vertex_translation->metal_function());

  if (pixel_translation && pixel_translation->metal_function()) {
    desc->setFragmentFunction(pixel_translation->metal_function());
  }

  for (uint32_t i = 0; i < 4; ++i) {
    desc->colorAttachments()->object(i)->setPixelFormat(
        formats.color_formats[i]);
  }
  desc->setDepthAttachmentPixelFormat(formats.depth_format);
  desc->setStencilAttachmentPixelFormat(formats.stencil_format);
  desc->setSampleCount(formats.sample_count);
  desc->setAlphaToCoverageEnabled(handle->pending_alpha_to_mask);

  ApplyBlendStateToDescriptor(desc->colorAttachments(),
                              handle->pending_normalized_color_mask,
                              handle->pending_blendcontrol);

  // Configure vertex fetch layout for MSC stage-in.
  const Shader& vertex_shader_ref = vertex_translation->shader();
  const auto& vertex_bindings = vertex_shader_ref.vertex_bindings();
  if (!ShaderUsesVertexFetch(vertex_shader_ref) && !vertex_bindings.empty()) {
    auto map_vertex_format =
        [](const ParsedVertexFetchInstruction::Attributes& attrs)
        -> MTL::VertexFormat {
      using xenos::VertexFormat;
      switch (attrs.data_format) {
        case VertexFormat::k_8_8_8_8:
          if (attrs.is_integer) {
            return attrs.is_signed ? MTL::VertexFormatChar4
                                   : MTL::VertexFormatUChar4;
          }
          return attrs.is_signed ? MTL::VertexFormatChar4Normalized
                                 : MTL::VertexFormatUChar4Normalized;
        case VertexFormat::k_2_10_10_10:
          return attrs.is_signed ? MTL::VertexFormatInt1010102Normalized
                                 : MTL::VertexFormatUInt1010102Normalized;
        case VertexFormat::k_10_11_11:
        case VertexFormat::k_11_11_10:
          return MTL::VertexFormatFloatRG11B10;
        case VertexFormat::k_16_16:
          if (attrs.is_integer) {
            return attrs.is_signed ? MTL::VertexFormatShort2
                                   : MTL::VertexFormatUShort2;
          }
          return attrs.is_signed ? MTL::VertexFormatShort2Normalized
                                 : MTL::VertexFormatUShort2Normalized;
        case VertexFormat::k_16_16_16_16:
          if (attrs.is_integer) {
            return attrs.is_signed ? MTL::VertexFormatShort4
                                   : MTL::VertexFormatUShort4;
          }
          return attrs.is_signed ? MTL::VertexFormatShort4Normalized
                                 : MTL::VertexFormatUShort4Normalized;
        case VertexFormat::k_16_16_FLOAT:
          return MTL::VertexFormatHalf2;
        case VertexFormat::k_16_16_16_16_FLOAT:
          return MTL::VertexFormatHalf4;
        case VertexFormat::k_32:
          if (attrs.is_integer) {
            return attrs.is_signed ? MTL::VertexFormatInt
                                   : MTL::VertexFormatUInt;
          }
          return MTL::VertexFormatFloat;
        case VertexFormat::k_32_32:
          if (attrs.is_integer) {
            return attrs.is_signed ? MTL::VertexFormatInt2
                                   : MTL::VertexFormatUInt2;
          }
          return MTL::VertexFormatFloat2;
        case VertexFormat::k_32_FLOAT:
          return MTL::VertexFormatFloat;
        case VertexFormat::k_32_32_FLOAT:
          return MTL::VertexFormatFloat2;
        case VertexFormat::k_32_32_32_FLOAT:
          return MTL::VertexFormatFloat3;
        case VertexFormat::k_32_32_32_32:
          if (attrs.is_integer) {
            return attrs.is_signed ? MTL::VertexFormatInt4
                                   : MTL::VertexFormatUInt4;
          }
          return MTL::VertexFormatFloat4;
        case VertexFormat::k_32_32_32_32_FLOAT:
          return MTL::VertexFormatFloat4;
        default:
          return MTL::VertexFormatInvalid;
      }
    };

    MTL::VertexDescriptor* vertex_desc = MTL::VertexDescriptor::alloc()->init();

    uint32_t attr_index = static_cast<uint32_t>(kIRStageInAttributeStartIndex);
    for (const auto& binding : vertex_bindings) {
      uint64_t buffer_index =
          kIRVertexBufferBindPoint + uint64_t(binding.binding_index);
      bool used_any_attribute = false;

      for (const auto& attr : binding.attributes) {
        MTL::VertexFormat fmt = map_vertex_format(attr.fetch_instr.attributes);
        if (fmt == MTL::VertexFormatInvalid) {
          ++attr_index;
          continue;
        }
        auto attr_desc = vertex_desc->attributes()->object(attr_index);
        attr_desc->setFormat(fmt);
        attr_desc->setOffset(
            static_cast<NS::UInteger>(attr.fetch_instr.attributes.offset * 4));
        attr_desc->setBufferIndex(static_cast<NS::UInteger>(buffer_index));
        used_any_attribute = true;
        ++attr_index;
      }

      if (used_any_attribute) {
        auto layout = vertex_desc->layouts()->object(buffer_index);
        layout->setStride(binding.stride_words * 4);
        layout->setStepFunction(MTL::VertexStepFunctionPerVertex);
        layout->setStepRate(1);
      }
    }

    desc->setVertexDescriptor(vertex_desc);
    vertex_desc->release();
  }

  {
    std::lock_guard<std::mutex> lock(pipeline_binary_archive_mutex_);
    if (pipeline_binary_archive_) {
      NS::Array* archives = NS::Array::array(pipeline_binary_archive_);
      desc->setBinaryArchives(archives);
      NS::Error* archive_error = nullptr;
      if (pipeline_binary_archive_->addRenderPipelineFunctions(
              desc, &archive_error)) {
        pipeline_binary_archive_dirty_ = true;
      }
    }
  }

  // Create pipeline state.
  NS::Error* error = nullptr;
  MTL::RenderPipelineState* pipeline =
      device_->newRenderPipelineState(desc, &error);
  desc->release();

  if (!pipeline) {
    if (error) {
      XELOGE("Failed to create pipeline state: {}",
             error->localizedDescription()->utf8String());
    } else {
      XELOGE("Failed to create pipeline state (unknown error)");
    }
    return nullptr;
  }

  return pipeline;
}

// ---------------------------------------------------------------------------
// CreationThread (background pipeline compilation)
// ---------------------------------------------------------------------------

void MetalPipelineCache::CreationThread(size_t thread_index) {
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

  while (true) {
    PipelineCreationRequest request;
    {
      std::unique_lock<std::mutex> lock(creation_request_lock_);
      while (creation_queue_.empty() && !creation_threads_shutdown_) {
        creation_request_cond_.wait(lock);
      }
      if (creation_threads_shutdown_ && creation_queue_.empty()) {
        break;
      }
      request = creation_queue_.top();
      creation_queue_.pop();
      ++creation_threads_busy_;
    }

    // Drain and recreate autorelease pool per pipeline.
    pool->drain();
    pool = NS::AutoreleasePool::alloc()->init();

    // Create the pipeline.
    PipelineHandle* handle = request.handle;
    MTL::RenderPipelineState* pipeline = CreatePipelineFromHandle(handle);
    handle->state.store(pipeline, std::memory_order_release);

    // Clear pending data.
    handle->pending_vertex_translation = nullptr;
    handle->pending_pixel_translation = nullptr;

    --creation_threads_busy_;
  }

  pool->drain();
}

// ---------------------------------------------------------------------------
// IsCreatingPipelines
// ---------------------------------------------------------------------------

bool MetalPipelineCache::IsCreatingPipelines() {
  if (creation_threads_.empty()) return false;
  std::lock_guard<std::mutex> lock(creation_request_lock_);
  return !creation_queue_.empty() || creation_threads_busy_ != 0;
}

// ---------------------------------------------------------------------------
// GetOrCreateGeometryPipelineState
// ---------------------------------------------------------------------------

MetalPipelineCache::GeometryPipelineState*
MetalPipelineCache::GetOrCreateGeometryPipelineState(
    MetalShader::MetalTranslation* vertex_translation,
    MetalShader::MetalTranslation* pixel_translation,
    GeometryShaderKey geometry_shader_key,
    const PipelineAttachmentFormats& attachment_formats,
    const PipelineRenderingKey& rendering_key) {
  if (!vertex_translation) {
    XELOGE("No valid vertex shader translation for geometry pipeline");
    return nullptr;
  }
  bool use_fallback_pixel_shader = (pixel_translation == nullptr);
  MTL::Library* pixel_library =
      use_fallback_pixel_shader ? nullptr : pixel_translation->metal_library();
  const char* pixel_function = use_fallback_pixel_shader
                                   ? nullptr
                                   : pixel_translation->function_name().c_str();
  if (use_fallback_pixel_shader) {
    if (!EnsureDepthOnlyPixelShader()) {
      XELOGE("Geometry pipeline: failed to create depth-only PS");
      return nullptr;
    }
    pixel_library = depth_only_pixel_library_;
    pixel_function = depth_only_pixel_function_name_.c_str();
  } else if (!pixel_library) {
    XELOGE("No valid pixel shader translation for geometry pipeline");
    return nullptr;
  }

  uint32_t sample_count = attachment_formats.sample_count;
  MTL::PixelFormat color_formats[4];
  for (uint32_t i = 0; i < 4; ++i) {
    color_formats[i] = attachment_formats.color_formats[i];
  }
  MTL::PixelFormat depth_format = attachment_formats.depth_format;
  MTL::PixelFormat stencil_format = attachment_formats.stencil_format;

  struct GeometryPipelineKey {
    const void* vs;
    const void* ps;
    uint32_t geometry_key;
    uint32_t sample_count;
    uint32_t depth_format;
    uint32_t stencil_format;
    uint32_t color_formats[4];
    uint32_t normalized_color_mask;
    uint32_t alpha_to_mask_enable;
    uint32_t blendcontrol[4];
  } key_data = {};

  key_data.vs = vertex_translation;
  key_data.ps = use_fallback_pixel_shader
                    ? static_cast<const void*>(pixel_library)
                    : static_cast<const void*>(pixel_translation);
  key_data.geometry_key = geometry_shader_key.key;
  key_data.sample_count = sample_count;
  key_data.depth_format = uint32_t(depth_format);
  key_data.stencil_format = uint32_t(stencil_format);
  for (uint32_t i = 0; i < 4; ++i) {
    key_data.color_formats[i] = uint32_t(color_formats[i]);
  }
  key_data.normalized_color_mask = rendering_key.normalized_color_mask;
  key_data.alpha_to_mask_enable = rendering_key.alpha_to_mask_enable;
  std::memcpy(key_data.blendcontrol, rendering_key.blendcontrol,
              sizeof(key_data.blendcontrol));
  uint64_t key = XXH3_64bits(&key_data, sizeof(key_data));

  auto it = geometry_pipeline_cache_.find(key);
  if (it != geometry_pipeline_cache_.end()) {
    return &it->second;
  }

  auto get_vertex_stage = [&]() -> GeometryVertexStageState* {
    auto vertex_it = geometry_vertex_stage_cache_.find(vertex_translation);
    if (vertex_it != geometry_vertex_stage_cache_.end()) {
      return &vertex_it->second;
    }

    std::vector<uint8_t> dxil_data = vertex_translation->dxil_data();
    if (dxil_data.empty()) {
      std::string dxil_error;
      if (!dxbc_to_dxil_converter_->Convert(
              vertex_translation->translated_binary(), dxil_data,
              &dxil_error)) {
        XELOGE("Geometry VS: DXBC to DXIL conversion failed: {}", dxil_error);
        return nullptr;
      }
    }

    struct InputAttribute {
      uint32_t input_slot = 0;
      uint32_t offset = 0;
      IRFormat format = IRFormatUnknown;
    };
    std::vector<InputAttribute> attribute_map;
    attribute_map.reserve(32);

    const Shader& vertex_shader_ref = vertex_translation->shader();
    const auto& vertex_bindings = vertex_shader_ref.vertex_bindings();
    uint32_t attr_index = 0;
    for (const auto& binding : vertex_bindings) {
      for (const auto& attr : binding.attributes) {
        if (attr_index >= 31) {
          break;
        }
        InputAttribute mapped = {};
        mapped.input_slot = static_cast<uint32_t>(binding.binding_index);
        mapped.offset =
            static_cast<uint32_t>(attr.fetch_instr.attributes.offset * 4);
        mapped.format = MapVertexFormatToIRFormat(attr.fetch_instr.attributes);
        attribute_map.push_back(mapped);
        ++attr_index;
      }
      if (attr_index >= 31) {
        break;
      }
    }

    IRInputTopology input_topology = IRInputTopologyUndefined;
    switch (geometry_shader_key.type) {
      case PipelineGeometryShader::kPointList:
        input_topology = IRInputTopologyPoint;
        break;
      case PipelineGeometryShader::kRectangleList:
        input_topology = IRInputTopologyTriangle;
        break;
      case PipelineGeometryShader::kQuadList:
        input_topology = IRInputTopologyUndefined;
        break;
      default:
        input_topology = IRInputTopologyUndefined;
        break;
    }
    MetalShaderConversionResult vertex_result;
    MetalShaderReflectionInfo vertex_reflection;

    if (!metal_shader_converter_->ConvertWithStageEx(
            MetalShaderStage::kVertex, dxil_data, vertex_result,
            &vertex_reflection, nullptr, nullptr, true,
            static_cast<int>(input_topology))) {
      XELOGE("Geometry VS: DXIL to Metal conversion failed: {}",
             vertex_result.error_message);
      return nullptr;
    }

    IRVersionedInputLayoutDescriptor input_layout = {};
    input_layout.version = IRInputLayoutDescriptorVersion_1;
    input_layout.desc_1_0.numElements = 0;
    std::vector<std::string> semantic_names_storage;
    if (!vertex_reflection.vertex_inputs.empty()) {
      semantic_names_storage.reserve(vertex_reflection.vertex_inputs.size());
      uint32_t element_count = 0;
      for (const auto& input : vertex_reflection.vertex_inputs) {
        if (element_count >= 31) {
          break;
        }
        if (input.attribute_index >= attribute_map.size()) {
          XELOGW("Geometry VS: vertex input {} out of range (max {})",
                 input.attribute_index, attribute_map.size());
          continue;
        }
        const InputAttribute& mapped = attribute_map[input.attribute_index];
        if (mapped.format == IRFormatUnknown) {
          XELOGW("Geometry VS: unknown IRFormat for vertex input {}",
                 input.attribute_index);
          continue;
        }
        std::string semantic_base = input.name;
        uint32_t semantic_index = 0;
        if (!semantic_base.empty()) {
          size_t digit_pos = semantic_base.size();
          while (digit_pos > 0 && std::isdigit(static_cast<unsigned char>(
                                      semantic_base[digit_pos - 1]))) {
            --digit_pos;
          }
          if (digit_pos < semantic_base.size()) {
            semantic_index = static_cast<uint32_t>(
                std::strtoul(semantic_base.c_str() + digit_pos, nullptr, 10));
            semantic_base.resize(digit_pos);
          }
        }
        if (semantic_base.empty()) {
          semantic_base = "TEXCOORD";
        }
        semantic_names_storage.push_back(std::move(semantic_base));
        input_layout.desc_1_0.semanticNames[element_count] =
            semantic_names_storage.back().c_str();
        IRInputElementDescriptor1& element =
            input_layout.desc_1_0.inputElementDescs[element_count];
        element.semanticIndex = semantic_index;
        element.format = mapped.format;
        element.inputSlot = mapped.input_slot;
        element.alignedByteOffset = mapped.offset;
        element.instanceDataStepRate = 0;
        element.inputSlotClass = IRInputClassificationPerVertexData;
        ++element_count;
      }
      input_layout.desc_1_0.numElements = element_count;
    }

    std::vector<uint8_t> stage_in_metallib;
    if (!metal_shader_converter_->ConvertWithStageEx(
            MetalShaderStage::kVertex, dxil_data, vertex_result,
            &vertex_reflection, &input_layout, &stage_in_metallib, true,
            static_cast<int>(input_topology))) {
      XELOGE("Geometry VS: DXIL to Metal conversion failed: {}",
             vertex_result.error_message);
      return nullptr;
    }
    if (stage_in_metallib.empty()) {
      XELOGE(
          "Geometry VS: Failed to synthesize stage-in function "
          "(vertex_inputs={}, output_size={})",
          vertex_reflection.vertex_input_count,
          vertex_reflection.vertex_output_size_in_bytes);
      return nullptr;
    }

    NS::Error* error = nullptr;
    dispatch_data_t vertex_data = dispatch_data_create(
        vertex_result.metallib_data.data(), vertex_result.metallib_data.size(),
        nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    MTL::Library* vertex_library = device_->newLibrary(vertex_data, &error);
    dispatch_release(vertex_data);
    if (!vertex_library) {
      XELOGE("Geometry VS: Failed to create Metal library: {}",
             error ? error->localizedDescription()->utf8String()
                   : "unknown error");
      return nullptr;
    }

    NS::Error* stage_in_error = nullptr;
    dispatch_data_t stage_in_data =
        dispatch_data_create(stage_in_metallib.data(), stage_in_metallib.size(),
                             nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    MTL::Library* stage_in_library =
        device_->newLibrary(stage_in_data, &stage_in_error);
    dispatch_release(stage_in_data);
    if (!stage_in_library) {
      XELOGE("Geometry VS: Failed to create stage-in library: {}",
             stage_in_error
                 ? stage_in_error->localizedDescription()->utf8String()
                 : "unknown error");
      vertex_library->release();
      return nullptr;
    }

    GeometryVertexStageState state;
    state.library = vertex_library;
    state.stage_in_library = stage_in_library;
    state.function_name = vertex_result.function_name;
    state.vertex_output_size_in_bytes =
        vertex_reflection.vertex_output_size_in_bytes;
    if (state.vertex_output_size_in_bytes == 0) {
      XELOGE(
          "Geometry VS: reflection returned zero output size "
          "(vertex_inputs={})",
          vertex_reflection.vertex_input_count);
    }

    auto [inserted_it, inserted] = geometry_vertex_stage_cache_.emplace(
        vertex_translation, std::move(state));
    return &inserted_it->second;
  };

  auto get_geometry_stage = [&]() -> GeometryShaderStageState* {
    auto geom_it = geometry_shader_stage_cache_.find(geometry_shader_key);
    if (geom_it != geometry_shader_stage_cache_.end()) {
      return &geom_it->second;
    }

    const std::vector<uint32_t>& dxbc_dwords =
        GetGeometryShader(geometry_shader_key);
    std::vector<uint8_t> dxbc_bytes(dxbc_dwords.size() * sizeof(uint32_t));
    std::memcpy(dxbc_bytes.data(), dxbc_dwords.data(), dxbc_bytes.size());

    std::vector<uint8_t> dxil_data;
    std::string dxil_error;
    if (!dxbc_to_dxil_converter_->Convert(dxbc_bytes, dxil_data, &dxil_error)) {
      XELOGE("Geometry GS: DXBC to DXIL conversion failed: {}", dxil_error);
      return nullptr;
    }

    IRInputTopology input_topology = IRInputTopologyUndefined;
    switch (geometry_shader_key.type) {
      case PipelineGeometryShader::kPointList:
        input_topology = IRInputTopologyPoint;
        break;
      case PipelineGeometryShader::kRectangleList:
        input_topology = IRInputTopologyTriangle;
        break;
      case PipelineGeometryShader::kQuadList:
        input_topology = IRInputTopologyUndefined;
        break;
      default:
        input_topology = IRInputTopologyUndefined;
        break;
    }
    MetalShaderConversionResult geometry_result;
    MetalShaderReflectionInfo geometry_reflection;
    if (!metal_shader_converter_->ConvertWithStageEx(
            MetalShaderStage::kGeometry, dxil_data, geometry_result,
            &geometry_reflection, nullptr, nullptr, true,
            static_cast<int>(input_topology))) {
      XELOGE("Geometry GS: DXIL to Metal conversion failed: {}",
             geometry_result.error_message);
      return nullptr;
    }
    if (!geometry_result.has_mesh_stage &&
        !geometry_result.has_geometry_stage) {
      XELOGE(
          "Geometry GS: MSC did not emit mesh or geometry stage (mesh={}, "
          "geometry={})",
          geometry_result.has_mesh_stage, geometry_result.has_geometry_stage);
      return nullptr;
    }
    if (!geometry_result.has_mesh_stage) {
      static bool mesh_missing_logged = false;
      if (!mesh_missing_logged) {
        mesh_missing_logged = true;
        XELOGW(
            "Geometry GS: MSC did not emit mesh stage; using geometry stage "
            "library");
      }
    }

    NS::Error* error = nullptr;
    dispatch_data_t geometry_data =
        dispatch_data_create(geometry_result.metallib_data.data(),
                             geometry_result.metallib_data.size(), nullptr,
                             DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    MTL::Library* geometry_library = device_->newLibrary(geometry_data, &error);
    dispatch_release(geometry_data);
    if (!geometry_library) {
      XELOGE("Geometry GS: Failed to create Metal library: {}",
             error ? error->localizedDescription()->utf8String()
                   : "unknown error");
      return nullptr;
    }

    GeometryShaderStageState state;
    state.library = geometry_library;
    state.function_name = geometry_result.function_name;
    state.max_input_primitives_per_mesh_threadgroup =
        geometry_reflection.gs_max_input_primitives_per_mesh_threadgroup;
    state.function_constants = geometry_reflection.function_constants;
    if (state.max_input_primitives_per_mesh_threadgroup == 0) {
      XELOGE("Geometry GS: reflection returned zero max input primitives");
    }

    auto [inserted_it, inserted] = geometry_shader_stage_cache_.emplace(
        geometry_shader_key, std::move(state));
    return &inserted_it->second;
  };

  GeometryVertexStageState* vertex_stage = get_vertex_stage();
  if (!vertex_stage || !vertex_stage->library ||
      !vertex_stage->stage_in_library) {
    return nullptr;
  }
  GeometryShaderStageState* geometry_stage = get_geometry_stage();
  if (!geometry_stage || !geometry_stage->library) {
    return nullptr;
  }

  MTL::MeshRenderPipelineDescriptor* desc =
      MTL::MeshRenderPipelineDescriptor::alloc()->init();

  for (uint32_t i = 0; i < 4; ++i) {
    desc->colorAttachments()->object(i)->setPixelFormat(color_formats[i]);
  }
  desc->setDepthAttachmentPixelFormat(depth_format);
  desc->setStencilAttachmentPixelFormat(stencil_format);
  desc->setRasterSampleCount(sample_count);
  desc->setAlphaToCoverageEnabled(key_data.alpha_to_mask_enable != 0);

  ApplyBlendStateToDescriptor(desc->colorAttachments(),
                              key_data.normalized_color_mask,
                              key_data.blendcontrol);
  if (!vertex_stage->vertex_output_size_in_bytes ||
      !geometry_stage->max_input_primitives_per_mesh_threadgroup) {
    XELOGE(
        "Geometry pipeline: invalid reflection (vs_output={}, gs_max_input={})",
        vertex_stage->vertex_output_size_in_bytes,
        geometry_stage->max_input_primitives_per_mesh_threadgroup);
    return nullptr;
  }

  IRGeometryEmulationPipelineDescriptor ir_desc = {};
  ir_desc.stageInLibrary = vertex_stage->stage_in_library;
  ir_desc.vertexLibrary = vertex_stage->library;
  ir_desc.vertexFunctionName = vertex_stage->function_name.c_str();
  ir_desc.geometryLibrary = geometry_stage->library;
  ir_desc.geometryFunctionName = geometry_stage->function_name.c_str();
  ir_desc.fragmentLibrary = pixel_library;
  ir_desc.fragmentFunctionName = pixel_function;
  ir_desc.basePipelineDescriptor = desc;
  ir_desc.pipelineConfig.gsVertexSizeInBytes =
      vertex_stage->vertex_output_size_in_bytes;
  ir_desc.pipelineConfig.gsMaxInputPrimitivesPerMeshThreadgroup =
      geometry_stage->max_input_primitives_per_mesh_threadgroup;

  NS::Error* error = nullptr;
  MTL::RenderPipelineState* pipeline =
      IRRuntimeNewGeometryEmulationPipeline(device_, &ir_desc, &error);
  desc->release();

  if (!pipeline) {
    XELOGE(
        "Failed to create geometry pipeline state: {}",
        error ? error->localizedDescription()->utf8String() : "unknown error");
    XELOGE(
        "Geometry pipeline details: vs_fn='{}' gs_fn='{}' ps_fn='{}' "
        "depth_format={} stencil_format={} samples={}",
        vertex_stage->function_name, geometry_stage->function_name,
        pixel_function ? pixel_function : "<null>", uint32_t(depth_format),
        uint32_t(stencil_format), sample_count);
    LogMetalErrorDetails("Geometry pipeline error", error);
    return nullptr;
  }

  GeometryPipelineState state;
  state.pipeline = pipeline;
  state.gs_vertex_size_in_bytes = ir_desc.pipelineConfig.gsVertexSizeInBytes;
  state.gs_max_input_primitives_per_mesh_threadgroup =
      ir_desc.pipelineConfig.gsMaxInputPrimitivesPerMeshThreadgroup;

  auto [inserted_it, inserted] =
      geometry_pipeline_cache_.emplace(key, std::move(state));
  return &inserted_it->second;
}

// ---------------------------------------------------------------------------
// GetOrCreateTessellationPipelineState
// ---------------------------------------------------------------------------

MetalPipelineCache::TessellationPipelineState*
MetalPipelineCache::GetOrCreateTessellationPipelineState(
    MetalShader::MetalTranslation* domain_translation,
    MetalShader::MetalTranslation* pixel_translation,
    const PrimitiveProcessor::ProcessingResult& primitive_processing_result,
    const PipelineAttachmentFormats& attachment_formats,
    const PipelineRenderingKey& rendering_key) {
  if (!domain_translation) {
    XELOGE("No valid domain shader translation for tessellation pipeline");
    return nullptr;
  }
  bool use_fallback_pixel_shader = (pixel_translation == nullptr);
  MTL::Library* pixel_library =
      use_fallback_pixel_shader ? nullptr : pixel_translation->metal_library();
  const char* pixel_function = use_fallback_pixel_shader
                                   ? nullptr
                                   : pixel_translation->function_name().c_str();
  if (use_fallback_pixel_shader) {
    if (!EnsureDepthOnlyPixelShader()) {
      XELOGE("Tessellation pipeline: failed to create depth-only PS");
      return nullptr;
    }
    pixel_library = depth_only_pixel_library_;
    pixel_function = depth_only_pixel_function_name_.c_str();
  } else if (!pixel_library) {
    XELOGE("No valid pixel shader translation for tessellation pipeline");
    return nullptr;
  }

  uint32_t sample_count = attachment_formats.sample_count;
  MTL::PixelFormat color_formats[4];
  for (uint32_t i = 0; i < 4; ++i) {
    color_formats[i] = attachment_formats.color_formats[i];
  }
  MTL::PixelFormat depth_format = attachment_formats.depth_format;
  MTL::PixelFormat stencil_format = attachment_formats.stencil_format;

  struct TessellationPipelineKey {
    const void* ds;
    const void* ps;
    uint32_t host_vs_type;
    uint32_t tessellation_mode;
    uint32_t host_prim;
    uint32_t sample_count;
    uint32_t depth_format;
    uint32_t stencil_format;
    uint32_t color_formats[4];
    uint32_t normalized_color_mask;
    uint32_t alpha_to_mask_enable;
    uint32_t blendcontrol[4];
  } key_data = {};

  key_data.ds = domain_translation;
  key_data.ps = use_fallback_pixel_shader
                    ? static_cast<const void*>(pixel_library)
                    : static_cast<const void*>(pixel_translation);
  key_data.host_vs_type =
      uint32_t(primitive_processing_result.host_vertex_shader_type);
  key_data.tessellation_mode =
      uint32_t(primitive_processing_result.tessellation_mode);
  key_data.host_prim =
      uint32_t(primitive_processing_result.host_primitive_type);
  key_data.sample_count = sample_count;
  key_data.depth_format = uint32_t(depth_format);
  key_data.stencil_format = uint32_t(stencil_format);
  for (uint32_t i = 0; i < 4; ++i) {
    key_data.color_formats[i] = uint32_t(color_formats[i]);
  }
  key_data.normalized_color_mask = rendering_key.normalized_color_mask;
  key_data.alpha_to_mask_enable = rendering_key.alpha_to_mask_enable;
  std::memcpy(key_data.blendcontrol, rendering_key.blendcontrol,
              sizeof(key_data.blendcontrol));
  uint64_t key = XXH3_64bits(&key_data, sizeof(key_data));

  auto it = tessellation_pipeline_cache_.find(key);
  if (it != tessellation_pipeline_cache_.end()) {
    return &it->second;
  }

  xenos::TessellationMode tessellation_mode =
      primitive_processing_result.tessellation_mode;

  auto get_vertex_stage = [&]() -> TessellationVertexStageState* {
    struct VertexStageKey {
      const void* shader;
      uint32_t tessellation_mode;
    } vertex_key = {domain_translation, uint32_t(tessellation_mode)};
    uint64_t vertex_key_hash = XXH3_64bits(&vertex_key, sizeof(vertex_key));
    auto vertex_it = tessellation_vertex_stage_cache_.find(vertex_key_hash);
    if (vertex_it != tessellation_vertex_stage_cache_.end()) {
      return &vertex_it->second;
    }

    const uint8_t* vs_bytes = nullptr;
    size_t vs_size = 0;
    if (tessellation_mode == xenos::TessellationMode::kAdaptive) {
      vs_bytes = ::tessellation_adaptive_vs;
      vs_size = sizeof(::tessellation_adaptive_vs);
    } else {
      vs_bytes = ::tessellation_indexed_vs;
      vs_size = sizeof(::tessellation_indexed_vs);
    }
    std::vector<uint8_t> dxbc_bytes(vs_bytes, vs_bytes + vs_size);
    std::vector<uint8_t> dxil_data;
    std::string dxil_error;
    if (!dxbc_to_dxil_converter_->Convert(dxbc_bytes, dxil_data, &dxil_error)) {
      XELOGE("Tessellation VS: DXBC to DXIL conversion failed: {}", dxil_error);
      return nullptr;
    }

    struct InputAttribute {
      uint32_t input_slot = 0;
      uint32_t offset = 0;
      IRFormat format = IRFormatUnknown;
    };
    std::vector<InputAttribute> attribute_map;
    attribute_map.reserve(32);

    const Shader& vertex_shader_ref = domain_translation->shader();
    const auto& vertex_bindings = vertex_shader_ref.vertex_bindings();
    uint32_t attr_index = 0;
    for (const auto& binding : vertex_bindings) {
      for (const auto& attr : binding.attributes) {
        if (attr_index >= 31) {
          break;
        }
        InputAttribute mapped = {};
        mapped.input_slot = static_cast<uint32_t>(binding.binding_index);
        mapped.offset =
            static_cast<uint32_t>(attr.fetch_instr.attributes.offset * 4);
        mapped.format = MapVertexFormatToIRFormat(attr.fetch_instr.attributes);
        attribute_map.push_back(mapped);
        ++attr_index;
      }
      if (attr_index >= 31) {
        break;
      }
    }

    IRInputTopology input_topology = IRInputTopologyUndefined;

    MetalShaderConversionResult vertex_result;
    MetalShaderReflectionInfo vertex_reflection;
    if (!metal_shader_converter_->ConvertWithStageEx(
            MetalShaderStage::kVertex, dxil_data, vertex_result,
            &vertex_reflection, nullptr, nullptr, true,
            static_cast<int>(input_topology))) {
      XELOGE("Tessellation VS: DXIL to Metal conversion failed: {}",
             vertex_result.error_message);
      return nullptr;
    }

    IRVersionedInputLayoutDescriptor input_layout = {};
    input_layout.version = IRInputLayoutDescriptorVersion_1;
    input_layout.desc_1_0.numElements = 0;
    std::vector<std::string> semantic_names_storage;
    if (!vertex_reflection.vertex_inputs.empty()) {
      semantic_names_storage.reserve(vertex_reflection.vertex_inputs.size());
      uint32_t element_count = 0;
      for (const auto& input : vertex_reflection.vertex_inputs) {
        if (element_count >= 31) {
          break;
        }
        if (input.attribute_index >= attribute_map.size()) {
          XELOGW("Tessellation VS: vertex input {} out of range (max {})",
                 input.attribute_index, attribute_map.size());
          continue;
        }
        const InputAttribute& mapped = attribute_map[input.attribute_index];
        if (mapped.format == IRFormatUnknown) {
          XELOGW("Tessellation VS: unknown IRFormat for vertex input {}",
                 input.attribute_index);
          continue;
        }
        std::string semantic_base = input.name;
        uint32_t semantic_index = 0;
        if (!semantic_base.empty()) {
          size_t digit_pos = semantic_base.size();
          while (digit_pos > 0 && std::isdigit(static_cast<unsigned char>(
                                      semantic_base[digit_pos - 1]))) {
            --digit_pos;
          }
          if (digit_pos < semantic_base.size()) {
            semantic_index = static_cast<uint32_t>(
                std::strtoul(semantic_base.c_str() + digit_pos, nullptr, 10));
            semantic_base.resize(digit_pos);
          }
        }
        if (semantic_base.empty()) {
          semantic_base = "TEXCOORD";
        }
        semantic_names_storage.push_back(std::move(semantic_base));
        input_layout.desc_1_0.semanticNames[element_count] =
            semantic_names_storage.back().c_str();
        IRInputElementDescriptor1& element =
            input_layout.desc_1_0.inputElementDescs[element_count];
        element.semanticIndex = semantic_index;
        element.format = mapped.format;
        element.inputSlot = mapped.input_slot;
        element.alignedByteOffset = mapped.offset;
        element.instanceDataStepRate = 0;
        element.inputSlotClass = IRInputClassificationPerVertexData;
        ++element_count;
      }
      input_layout.desc_1_0.numElements = element_count;
    }

    std::vector<uint8_t> stage_in_metallib;
    if (!metal_shader_converter_->ConvertWithStageEx(
            MetalShaderStage::kVertex, dxil_data, vertex_result,
            &vertex_reflection, &input_layout, &stage_in_metallib, true,
            static_cast<int>(input_topology))) {
      XELOGE("Tessellation VS: DXIL to Metal conversion failed: {}",
             vertex_result.error_message);
      return nullptr;
    }
    if (stage_in_metallib.empty()) {
      XELOGE("Tessellation VS: Failed to synthesize stage-in function");
      return nullptr;
    }

    NS::Error* error = nullptr;
    dispatch_data_t vertex_data = dispatch_data_create(
        vertex_result.metallib_data.data(), vertex_result.metallib_data.size(),
        nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    MTL::Library* vertex_library = device_->newLibrary(vertex_data, &error);
    dispatch_release(vertex_data);
    if (!vertex_library) {
      XELOGE("Tessellation VS: Failed to create Metal library: {}",
             error ? error->localizedDescription()->utf8String()
                   : "unknown error");
      return nullptr;
    }

    NS::Error* stage_in_error = nullptr;
    dispatch_data_t stage_in_data =
        dispatch_data_create(stage_in_metallib.data(), stage_in_metallib.size(),
                             nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    MTL::Library* stage_in_library =
        device_->newLibrary(stage_in_data, &stage_in_error);
    dispatch_release(stage_in_data);
    if (!stage_in_library) {
      XELOGE("Tessellation VS: Failed to create stage-in library: {}",
             stage_in_error
                 ? stage_in_error->localizedDescription()->utf8String()
                 : "unknown error");
      vertex_library->release();
      return nullptr;
    }

    TessellationVertexStageState state;
    state.library = vertex_library;
    state.stage_in_library = stage_in_library;
    state.function_name = vertex_result.function_name;
    state.vertex_output_size_in_bytes =
        vertex_reflection.vertex_output_size_in_bytes;
    if (state.vertex_output_size_in_bytes == 0) {
      XELOGE("Tessellation VS: reflection returned zero output size");
    }

    auto [inserted_it, inserted] = tessellation_vertex_stage_cache_.emplace(
        vertex_key_hash, std::move(state));
    return &inserted_it->second;
  };

  auto get_hull_stage = [&]() -> TessellationHullStageState* {
    struct HullStageKey {
      uint32_t host_vs_type;
      uint32_t tessellation_mode;
    } hull_key = {uint32_t(primitive_processing_result.host_vertex_shader_type),
                  uint32_t(tessellation_mode)};
    uint64_t hull_key_hash = XXH3_64bits(&hull_key, sizeof(hull_key));
    auto hull_it = tessellation_hull_stage_cache_.find(hull_key_hash);
    if (hull_it != tessellation_hull_stage_cache_.end()) {
      return &hull_it->second;
    }

    const uint8_t* hs_bytes = nullptr;
    size_t hs_size = 0;
    switch (tessellation_mode) {
      case xenos::TessellationMode::kDiscrete:
        switch (primitive_processing_result.host_vertex_shader_type) {
          case Shader::HostVertexShaderType::kTriangleDomainCPIndexed:
            hs_bytes = ::discrete_triangle_3cp_hs;
            hs_size = sizeof(::discrete_triangle_3cp_hs);
            break;
          case Shader::HostVertexShaderType::kTriangleDomainPatchIndexed:
            hs_bytes = ::discrete_triangle_1cp_hs;
            hs_size = sizeof(::discrete_triangle_1cp_hs);
            break;
          case Shader::HostVertexShaderType::kQuadDomainCPIndexed:
            hs_bytes = ::discrete_quad_4cp_hs;
            hs_size = sizeof(::discrete_quad_4cp_hs);
            break;
          case Shader::HostVertexShaderType::kQuadDomainPatchIndexed:
            hs_bytes = ::discrete_quad_1cp_hs;
            hs_size = sizeof(::discrete_quad_1cp_hs);
            break;
          default:
            XELOGE(
                "Tessellation HS: unsupported host vertex shader type {}",
                uint32_t(primitive_processing_result.host_vertex_shader_type));
            return nullptr;
        }
        break;
      case xenos::TessellationMode::kContinuous:
        switch (primitive_processing_result.host_vertex_shader_type) {
          case Shader::HostVertexShaderType::kTriangleDomainCPIndexed:
            hs_bytes = ::continuous_triangle_3cp_hs;
            hs_size = sizeof(::continuous_triangle_3cp_hs);
            break;
          case Shader::HostVertexShaderType::kTriangleDomainPatchIndexed:
            hs_bytes = ::continuous_triangle_1cp_hs;
            hs_size = sizeof(::continuous_triangle_1cp_hs);
            break;
          case Shader::HostVertexShaderType::kQuadDomainCPIndexed:
            hs_bytes = ::continuous_quad_4cp_hs;
            hs_size = sizeof(::continuous_quad_4cp_hs);
            break;
          case Shader::HostVertexShaderType::kQuadDomainPatchIndexed:
            hs_bytes = ::continuous_quad_1cp_hs;
            hs_size = sizeof(::continuous_quad_1cp_hs);
            break;
          default:
            XELOGE(
                "Tessellation HS: unsupported host vertex shader type {}",
                uint32_t(primitive_processing_result.host_vertex_shader_type));
            return nullptr;
        }
        break;
      case xenos::TessellationMode::kAdaptive:
        switch (primitive_processing_result.host_vertex_shader_type) {
          case Shader::HostVertexShaderType::kTriangleDomainPatchIndexed:
            hs_bytes = ::adaptive_triangle_hs;
            hs_size = sizeof(::adaptive_triangle_hs);
            break;
          case Shader::HostVertexShaderType::kQuadDomainPatchIndexed:
            hs_bytes = ::adaptive_quad_hs;
            hs_size = sizeof(::adaptive_quad_hs);
            break;
          default:
            XELOGE(
                "Tessellation HS: unsupported host vertex shader type {}",
                uint32_t(primitive_processing_result.host_vertex_shader_type));
            return nullptr;
        }
        break;
      default:
        XELOGE("Tessellation HS: unsupported tessellation mode {}",
               uint32_t(tessellation_mode));
        return nullptr;
    }

    std::vector<uint8_t> dxbc_bytes(hs_bytes, hs_bytes + hs_size);
    std::vector<uint8_t> dxil_data;
    std::string dxil_error;
    if (!dxbc_to_dxil_converter_->Convert(dxbc_bytes, dxil_data, &dxil_error)) {
      XELOGE("Tessellation HS: DXBC to DXIL conversion failed: {}", dxil_error);
      return nullptr;
    }

    MetalShaderConversionResult hull_result;
    MetalShaderReflectionInfo hull_reflection;
    if (!metal_shader_converter_->ConvertWithStageEx(
            MetalShaderStage::kHull, dxil_data, hull_result, &hull_reflection,
            nullptr, nullptr, true,
            static_cast<int>(IRInputTopologyUndefined))) {
      XELOGE("Tessellation HS: DXIL to Metal conversion failed: {}",
             hull_result.error_message);
      return nullptr;
    }

    NS::Error* error = nullptr;
    dispatch_data_t hull_data = dispatch_data_create(
        hull_result.metallib_data.data(), hull_result.metallib_data.size(),
        nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    MTL::Library* hull_library = device_->newLibrary(hull_data, &error);
    dispatch_release(hull_data);
    if (!hull_library) {
      XELOGE("Tessellation HS: Failed to create Metal library: {}",
             error ? error->localizedDescription()->utf8String()
                   : "unknown error");
      return nullptr;
    }

    TessellationHullStageState state;
    state.library = hull_library;
    state.function_name = hull_result.function_name;
    state.reflection = hull_reflection;
    if (!state.reflection.has_hull_info) {
      XELOGE("Tessellation HS: reflection missing hull info");
    }

    auto [inserted_it, inserted] =
        tessellation_hull_stage_cache_.emplace(hull_key_hash, std::move(state));
    return &inserted_it->second;
  };

  auto get_domain_stage = [&]() -> TessellationDomainStageState* {
    uint64_t domain_key =
        XXH3_64bits(&domain_translation, sizeof(domain_translation));
    auto domain_it = tessellation_domain_stage_cache_.find(domain_key);
    if (domain_it != tessellation_domain_stage_cache_.end()) {
      return &domain_it->second;
    }

    std::vector<uint8_t> dxil_data = domain_translation->dxil_data();
    if (dxil_data.empty()) {
      std::string dxil_error;
      if (!dxbc_to_dxil_converter_->Convert(
              domain_translation->translated_binary(), dxil_data,
              &dxil_error)) {
        XELOGE("Tessellation DS: DXBC to DXIL conversion failed: {}",
               dxil_error);
        return nullptr;
      }
    }

    MetalShaderConversionResult domain_result;
    MetalShaderReflectionInfo domain_reflection;
    if (!metal_shader_converter_->ConvertWithStageEx(
            MetalShaderStage::kDomain, dxil_data, domain_result,
            &domain_reflection, nullptr, nullptr, true,
            static_cast<int>(IRInputTopologyUndefined))) {
      XELOGE("Tessellation DS: DXIL to Metal conversion failed: {}",
             domain_result.error_message);
      return nullptr;
    }

    NS::Error* error = nullptr;
    dispatch_data_t domain_data = dispatch_data_create(
        domain_result.metallib_data.data(), domain_result.metallib_data.size(),
        nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    MTL::Library* domain_library = device_->newLibrary(domain_data, &error);
    dispatch_release(domain_data);
    if (!domain_library) {
      XELOGE("Tessellation DS: Failed to create Metal library: {}",
             error ? error->localizedDescription()->utf8String()
                   : "unknown error");
      return nullptr;
    }

    TessellationDomainStageState state;
    state.library = domain_library;
    state.function_name = domain_result.function_name;
    state.reflection = domain_reflection;
    if (!state.reflection.has_domain_info) {
      XELOGE("Tessellation DS: reflection missing domain info");
    }

    auto [inserted_it, inserted] =
        tessellation_domain_stage_cache_.emplace(domain_key, std::move(state));
    return &inserted_it->second;
  };

  TessellationVertexStageState* vertex_stage = get_vertex_stage();
  if (!vertex_stage || !vertex_stage->library ||
      !vertex_stage->stage_in_library) {
    return nullptr;
  }
  TessellationHullStageState* hull_stage = get_hull_stage();
  if (!hull_stage || !hull_stage->library) {
    return nullptr;
  }
  TessellationDomainStageState* domain_stage = get_domain_stage();
  if (!domain_stage || !domain_stage->library) {
    return nullptr;
  }

  IRRuntimeTessellatorOutputPrimitive output_primitive =
      IRRuntimeTessellatorOutputUndefined;
  switch (hull_stage->reflection.hs_tessellator_output_primitive) {
    case IRRuntimeTessellatorOutputPoint:
      output_primitive = IRRuntimeTessellatorOutputPoint;
      break;
    case IRRuntimeTessellatorOutputLine:
      output_primitive = IRRuntimeTessellatorOutputLine;
      break;
    case IRRuntimeTessellatorOutputTriangleCW:
      output_primitive = IRRuntimeTessellatorOutputTriangleCW;
      break;
    case IRRuntimeTessellatorOutputTriangleCCW:
      output_primitive = IRRuntimeTessellatorOutputTriangleCCW;
      break;
    default:
      XELOGE("Tessellation pipeline: unsupported tessellator output {}",
             hull_stage->reflection.hs_tessellator_output_primitive);
      return nullptr;
  }

  IRRuntimePrimitiveType geometry_primitive = IRRuntimePrimitiveTypeTriangle;
  const char* geometry_function = kIRTrianglePassthroughGeometryShader;
  switch (output_primitive) {
    case IRRuntimeTessellatorOutputPoint:
      geometry_primitive = IRRuntimePrimitiveTypePoint;
      geometry_function = kIRPointPassthroughGeometryShader;
      break;
    case IRRuntimeTessellatorOutputLine:
      geometry_primitive = IRRuntimePrimitiveTypeLine;
      geometry_function = kIRLinePassthroughGeometryShader;
      break;
    case IRRuntimeTessellatorOutputTriangleCW:
    case IRRuntimeTessellatorOutputTriangleCCW:
      geometry_primitive = IRRuntimePrimitiveTypeTriangle;
      geometry_function = kIRTrianglePassthroughGeometryShader;
      break;
    default:
      break;
  }

  if (!IRRuntimeValidateTessellationPipeline(
          output_primitive, geometry_primitive,
          hull_stage->reflection.hs_output_control_point_size,
          domain_stage->reflection.ds_input_control_point_size,
          hull_stage->reflection.hs_patch_constants_size,
          domain_stage->reflection.ds_patch_constants_size,
          hull_stage->reflection.hs_output_control_point_count,
          domain_stage->reflection.ds_input_control_point_count)) {
    XELOGE("Tessellation pipeline: validation failed for HS/DS pairing");
    return nullptr;
  }

  MTL::MeshRenderPipelineDescriptor* desc =
      MTL::MeshRenderPipelineDescriptor::alloc()->init();
  for (uint32_t i = 0; i < 4; ++i) {
    desc->colorAttachments()->object(i)->setPixelFormat(color_formats[i]);
  }
  desc->setDepthAttachmentPixelFormat(depth_format);
  desc->setStencilAttachmentPixelFormat(stencil_format);
  desc->setRasterSampleCount(sample_count);
  desc->setAlphaToCoverageEnabled(key_data.alpha_to_mask_enable != 0);

  ApplyBlendStateToDescriptor(desc->colorAttachments(),
                              key_data.normalized_color_mask,
                              key_data.blendcontrol);
  IRGeometryTessellationEmulationPipelineDescriptor ir_desc = {};
  ir_desc.stageInLibrary = vertex_stage->stage_in_library;
  ir_desc.vertexLibrary = vertex_stage->library;
  ir_desc.vertexFunctionName = vertex_stage->function_name.c_str();
  ir_desc.hullLibrary = hull_stage->library;
  ir_desc.hullFunctionName = hull_stage->function_name.c_str();
  ir_desc.domainLibrary = domain_stage->library;
  ir_desc.domainFunctionName = domain_stage->function_name.c_str();
  ir_desc.geometryLibrary = nullptr;
  ir_desc.geometryFunctionName = geometry_function;
  ir_desc.fragmentLibrary = pixel_library;
  ir_desc.fragmentFunctionName = pixel_function;
  ir_desc.basePipelineDescriptor = desc;
  ir_desc.pipelineConfig.outputPrimitiveType = output_primitive;
  ir_desc.pipelineConfig.vsOutputSizeInBytes =
      vertex_stage->vertex_output_size_in_bytes;
  ir_desc.pipelineConfig.gsMaxInputPrimitivesPerMeshThreadgroup =
      domain_stage->reflection.ds_max_input_prims_per_mesh_threadgroup;
  ir_desc.pipelineConfig.hsMaxPatchesPerObjectThreadgroup =
      hull_stage->reflection.hs_max_patches_per_object_threadgroup;
  ir_desc.pipelineConfig.hsInputControlPointCount =
      hull_stage->reflection.hs_input_control_point_count;
  ir_desc.pipelineConfig.hsMaxObjectThreadsPerThreadgroup =
      hull_stage->reflection.hs_max_object_threads_per_patch;
  ir_desc.pipelineConfig.hsMaxTessellationFactor =
      hull_stage->reflection.hs_max_tessellation_factor;
  ir_desc.pipelineConfig.gsInstanceCount = 1;

  if (!ir_desc.pipelineConfig.vsOutputSizeInBytes ||
      !ir_desc.pipelineConfig.gsMaxInputPrimitivesPerMeshThreadgroup ||
      !ir_desc.pipelineConfig.hsMaxPatchesPerObjectThreadgroup ||
      !ir_desc.pipelineConfig.hsInputControlPointCount ||
      !ir_desc.pipelineConfig.hsMaxObjectThreadsPerThreadgroup) {
    XELOGE(
        "Tessellation pipeline: invalid reflection values (vs_output={}, "
        "gs_max_input={}, hs_patches={}, hs_cp_count={}, hs_threads={})",
        ir_desc.pipelineConfig.vsOutputSizeInBytes,
        ir_desc.pipelineConfig.gsMaxInputPrimitivesPerMeshThreadgroup,
        ir_desc.pipelineConfig.hsMaxPatchesPerObjectThreadgroup,
        ir_desc.pipelineConfig.hsInputControlPointCount,
        ir_desc.pipelineConfig.hsMaxObjectThreadsPerThreadgroup);
    desc->release();
    return nullptr;
  }

  NS::Error* error = nullptr;
  MTL::RenderPipelineState* pipeline =
      IRRuntimeNewGeometryTessellationEmulationPipeline(device_, &ir_desc,
                                                        &error);
  desc->release();
  if (!pipeline) {
    XELOGE(
        "Failed to create tessellation pipeline state: {}",
        error ? error->localizedDescription()->utf8String() : "unknown error");
    return nullptr;
  }

  TessellationPipelineState state;
  state.pipeline = pipeline;
  state.config = ir_desc.pipelineConfig;
  state.primitive = geometry_primitive;

  auto [inserted_it, inserted] =
      tessellation_pipeline_cache_.emplace(key, std::move(state));
  return &inserted_it->second;
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
