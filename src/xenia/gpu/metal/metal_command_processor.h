/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_COMMAND_PROCESSOR_H_
#define XENIA_GPU_METAL_METAL_COMMAND_PROCESSOR_H_

#include <dispatch/dispatch.h>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "xenia/base/platform.h"
#include "xenia/base/string_buffer.h"
#include "xenia/gpu/command_processor.h"
#include "xenia/gpu/draw_util.h"
#include "xenia/gpu/dxbc_shader_translator.h"
#include "xenia/gpu/metal/dxbc_to_dxil_converter.h"
#include "xenia/gpu/metal/metal_geometry_shader.h"
#include "xenia/gpu/metal/metal_pipeline_cache.h"
#include "xenia/gpu/metal/metal_primitive_processor.h"
#include "xenia/gpu/metal/metal_render_target_cache.h"
#include "xenia/gpu/metal/metal_shader.h"
#include "xenia/gpu/metal/metal_shader_converter.h"
#include "xenia/gpu/metal/metal_shared_memory.h"
#include "xenia/gpu/metal/metal_texture_cache.h"
#include "xenia/gpu/metal/metal_upload_buffer_pool.h"
// clang-format off
// Must come after metal_texture_cache.h which includes Metal.hpp
#include "third_party/metal-shader-converter/include/metal_irconverter_runtime.h"
// clang-format on
#include "xenia/ui/metal/metal_api.h"
#include "xenia/ui/metal/metal_provider.h"

namespace MTL {
class Heap;
class SharedEvent;
}  // namespace MTL

namespace xe {
namespace gpu {
namespace metal {

class MetalGraphicsSystem;

class MetalCommandProcessor : public CommandProcessor {
 protected:
#define OVERRIDING_BASE_CMDPROCESSOR
#include "../pm4_command_processor_declare.h"
#undef OVERRIDING_BASE_CMDPROCESSOR

 public:
  explicit MetalCommandProcessor(MetalGraphicsSystem* graphics_system,
                                 kernel::KernelState* kernel_state);
  ~MetalCommandProcessor();

  void TracePlaybackWroteMemory(uint32_t base_ptr, uint32_t length) override;
  void RestoreEdramSnapshot(const void* snapshot) override;
  void ClearCaches() override;
  void InvalidateGpuMemory() override;
  void ClearReadbackBuffers() override;

  // ---------------------------------------------------------------------------
  // Trace replay resolve protection.
  //
  // During trace playback the trace player replays MemoryRead commands that
  // blindly overwrite guest physical memory.  When the Metal backend has
  // already resolved (IssueCopy) into a region, those writes would clobber
  // live GPU-produced data with stale trace-file contents.
  //
  // TraceResolveGuard tracks resolved regions within a frame so the trace
  // player (or any other caller) can query whether a write should be skipped.
  // The guard is cleared at every frame boundary (IssueSwap /
  // RestoreEdramSnapshot).  It carries no state that participates in live
  // game rendering.
  // ---------------------------------------------------------------------------
  class TraceResolveGuard {
   public:
    // Record a resolved memory region (called from IssueCopy).
    void Mark(uint32_t base_ptr, uint32_t length);

    // Return true if the range overlaps any previously resolved region.
    bool IsResolved(uint32_t base_ptr, uint32_t length) const;

    // Drop all tracked ranges (frame boundary).
    void Clear();

   private:
    struct ResolvedRange {
      uint32_t base;
      uint32_t length;
    };
    std::vector<ResolvedRange> ranges_;
  };

  TraceResolveGuard& trace_resolve_guard() { return trace_resolve_guard_; }
  const TraceResolveGuard& trace_resolve_guard() const {
    return trace_resolve_guard_;
  }

  ui::metal::MetalProvider& GetMetalProvider() const;

  // Get the Metal device and command queue
  MTL::Device* GetMetalDevice() const { return device_; }
  MTL::CommandQueue* GetMetalCommandQueue() const { return command_queue_; }
  MTL::CommandBuffer* GetCurrentCommandBuffer() const {
    return current_command_buffer_;
  }

  // Submission coordination helpers — callers use these to query or obtain
  // command buffers for transfer/upload work without reaching into internal
  // command-processor state.
  bool HasActiveSubmission() const {
    return current_command_buffer_ != nullptr;
  }
  // Returns true when upload/transfer work can be encoded onto the current
  // submission's command buffer without a standalone detour.  This is the
  // case when a command buffer exists but no render encoder is open.
  bool CanJoinActiveSubmissionForTransfer() const {
    return current_command_buffer_ != nullptr &&
           current_render_encoder_ == nullptr;
  }
  // Returns a command buffer suitable for transfer (blit/compute) work.
  // If a render encoder is active it is ended first; if no command buffer
  // exists one is created.  Returns nullptr on failure.
  MTL::CommandBuffer* RequestTransferCommandBuffer();

  // Standalone (detached) transfer command-buffer helpers.
  // These create command buffers that are independent of the active submission
  // and are used by caches for upload/transfer work that cannot join the
  // current command buffer.  The returned CB is retained; ownership transfers
  // back via CommitStandaloneAsync or CommitStandaloneAndWait.
  MTL::CommandBuffer* CreateStandaloneTransferCommandBuffer(const char* label);
  // Commit a standalone command buffer asynchronously (fire-and-forget).
  // The CB is released via a completion handler.
  void CommitStandaloneAsync(MTL::CommandBuffer* cmd);
  // Commit a standalone command buffer synchronously and wait for completion.
  // The CB is released before returning.
  void CommitStandaloneAndWait(MTL::CommandBuffer* cmd);

  uint64_t GetCurrentSubmission() const;
  uint64_t GetCompletedSubmission() const;
  uint64_t GetLatestSubmissionStarted() const { return submission_current_; }
  MTL::CommandBuffer* EnsureCommandBuffer();
  void EndRenderEncoder();
  void ResetRenderEncoderResourceUsage();
  void UseRenderEncoderResource(MTL::Resource* resource,
                                MTL::ResourceUsage usage);
  void EnsureCommandBufferAutoreleasePool();
  void DrainCommandBufferAutoreleasePool();

  // Force issue a swap to push render target to presenter (for trace dumps)
  void ForceIssueSwap();
  bool HasSeenSwap() const { return saw_swap_; }
  void SetSwapDestSwap(uint32_t dest_base, bool swap);
  bool ConsumeSwapDestSwap(uint32_t dest_base, bool* swap_out);

  MetalSharedMemory* shared_memory() const { return shared_memory_.get(); }
  MetalRenderTargetCache* render_target_cache() const {
    return render_target_cache_.get();
  }
  MetalTextureCache* texture_cache() const { return texture_cache_.get(); }

  // Persistent bindless descriptor heap allocators.
  uint32_t AllocateViewBindlessIndex();
  void ReleaseViewBindlessIndex(uint32_t index);
  void RetireViewBindlessIndex(uint32_t index);
  uint32_t GetViewBindlessHeapAvailableCount() const;
  uint32_t AllocateSamplerBindlessIndex();
  void ReleaseSamplerBindlessIndex(uint32_t index);
  IRDescriptorTableEntry* GetViewBindlessHeapEntry(uint32_t index);
  IRDescriptorTableEntry* GetSamplerBindlessHeapEntry(uint32_t index);

  // Resolve ordering policy — controls command-buffer boundary placement
  // around resolve (IssueCopy) operations.  The baseline policy uses
  // submission boundaries for GPU write visibility because Metal lacks
  // D3D12's explicit UAV barrier model.  The strict path will introduce
  // additional policies (e.g. on-tile resolve) that override this.
  enum class ResolveOrderingPolicy {
    // Current safe baseline: end the command buffer before and after every
    // resolve to ensure GPU write visibility at submission boundaries.
    // This is correct but creates excess command-buffer submissions.
    kSubmissionBoundary,

    // Future: strict path uses on-tile resolve with no separate submission.
    // kOnTileResolve,
  };

 protected:
  bool SetupContext() override;
  void ShutdownContext() override;
  void InitializeShaderStorage(
      const std::filesystem::path& cache_root, uint32_t title_id, bool blocking,
      std::function<void()> completion_callback = nullptr) override;

  // Flush pending GPU work before entering wait state.
  // This ensures Metal command buffers are submitted and completed before
  // the autorelease pool is drained, preventing hangs from deferred
  // deallocation.
  void PrepareForWait() override;

  // Use base class WriteRegister - don't override with empty implementation!
  // The base class stores values in register_file_->values[] which we need.
  void OnPrimaryBufferEnd() override;
  void OnGammaRamp256EntryTableValueWritten() override;
  void OnGammaRampPWLValueWritten() override;

  void IssueSwap(uint32_t frontbuffer_ptr, uint32_t frontbuffer_width,
                 uint32_t frontbuffer_height) override;

  Shader* LoadShader(xenos::ShaderType shader_type, uint32_t guest_address,
                     const uint32_t* host_address,
                     uint32_t dword_count) override;

  bool IssueDraw(xenos::PrimitiveType primitive_type, uint32_t index_count,
                 IndexBufferInfo* index_buffer_info,
                 bool major_mode_explicit) override;
  bool IssueCopy() override;
  void WriteRegister(uint32_t index, uint32_t value) override;

  // ===========================================================================
  // Host render backend boundary.
  //
  // The methods below define the host-specific draw, copy/resolve, and
  // transfer entry points that IssueDraw / IssueCopy delegate to.  They are
  // virtual so a future strict-path backend can override them without
  // touching the guest-facing command processor logic.
  //
  // Draw path:   UploadConstants  -> PopulateBindlessTables -> DispatchDraw
  // Copy path:   BeginResolveOrdering -> Resolve -> EndResolveOrdering
  // Transfer:    MetalRenderTargetCache::PerformTransfersAndResolveClears
  //              (sole transfer execution entry point; called from both the
  //              draw path via Update and the copy path via Resolve -- no
  //              transfer operations bypass the render target cache)
  // ===========================================================================

  // Per-draw uniform buffer coordinates passed between IssueDraw sub-methods.
  struct UniformBufferInfo {
    MTL::Buffer* vs_buf = nullptr;
    NS::UInteger vs_off = 0;
    uint64_t vs_gpu = 0;
    MTL::Buffer* ps_buf = nullptr;
    NS::UInteger ps_off = 0;
    uint64_t ps_gpu = 0;
  };

  // Vertex binding range for stage-in / geometry emulation.
  struct VertexBindingRange {
    uint32_t binding_index = 0;
    uint32_t offset = 0;
    uint32_t length = 0;
    uint32_t stride = 0;
  };

  // Host draw path — upload per-draw constant buffers (system constants,
  // float/bool/loop/fetch constants, descriptor indices) and apply viewport,
  // scissor, rasterizer, depth-stencil, and blend state.
  virtual bool UploadConstants(
      const RegisterFile& regs, Shader* vertex_shader, Shader* pixel_shader,
      MetalShader* metal_vertex_shader, MetalShader* metal_pixel_shader,
      bool shared_memory_is_uav,
      const PrimitiveProcessor::ProcessingResult& primitive_processing_result,
      uint32_t used_texture_mask, uint32_t normalized_color_mask,
      UniformBufferInfo& uniforms_out);

  // Host draw path — build the per-draw bindless descriptor tables (top-level
  // argument buffer, CBV table) and bind them plus the heap buffers to the
  // render encoder.
  virtual bool PopulateBindlessTables(MetalShader* metal_vertex_shader,
                                      MetalShader* metal_pixel_shader,
                                      bool shared_memory_is_uav,
                                      MTL::ResourceUsage shared_memory_usage,
                                      bool use_geometry_emulation,
                                      bool use_tessellation_emulation,
                                      const UniformBufferInfo& uniforms);

  // Host draw path — bind vertex buffers and dispatch the actual draw call
  // (tessellation, geometry emulation, or standard path), then track
  // memexport writes.
  virtual bool DispatchDraw(
      const RegisterFile& regs,
      const PrimitiveProcessor::ProcessingResult& primitive_processing_result,
      bool use_tessellation_emulation,
      MetalPipelineCache::TessellationPipelineState*
          tessellation_pipeline_state,
      bool use_geometry_emulation,
      MetalPipelineCache::GeometryPipelineState* geometry_pipeline_state,
      bool shared_memory_is_uav, MTL::ResourceUsage shared_memory_usage,
      bool memexport_used, bool uses_vertex_fetch,
      const std::vector<Shader::VertexBinding>& vb_bindings,
      const VertexBindingRange* vertex_ranges, uint32_t vertex_range_count,
      IndexBufferInfo* index_buffer_info);

  // Host copy/resolve path — enforce the active ResolveOrderingPolicy
  // around resolve (IssueCopy) work.
  // Called before resolve work begins.  Ends the render encoder, applies
  // the pre-resolve boundary policy, ensures a command buffer, and returns
  // it.  Returns nullptr on failure.
  virtual MTL::CommandBuffer* BeginResolveOrdering();
  // Called after resolve work completes.  Applies the post-resolve boundary
  // policy.
  virtual void EndResolveOrdering();

 private:
  // Command buffer management
  void FlushCommandBufferAndWait(uint64_t timeout_ns, const char* context);
  void BeginCommandBuffer();
  void EndCommandBuffer();
  bool CanEndSubmissionImmediately();
  void WaitForPendingCompletionHandlers();
  void ProcessCompletedSubmissions();

  void UseRenderEncoderAttachmentHeaps(MTL::RenderPassDescriptor* descriptor);
  void UseRenderEncoderHeap(MTL::Heap* heap);
  uint64_t GetBindlessDescriptorRetirementSubmission() const;
  void FreeViewBindlessIndexNow(uint32_t index);
  void FreeSamplerBindlessIndexNow(uint32_t index);

  // Fixed-function depth/stencil state (mirrors Vulkan/D3D12 dynamic state).
  void ApplyDepthStencilState(bool primitive_polygonal,
                              reg::RB_DEPTHCONTROL normalized_depth_control);
  void ApplyRasterizerState(bool primitive_polygonal);

  // Constants for the MSC path.
  static constexpr size_t kStageCount = 2;  // Vertex + pixel.
  static constexpr size_t kNullBufferSize = 4096;
  static constexpr size_t kCbvSizeBytes = 4096;
  static constexpr size_t kUniformsBytesPerTable = 6 * kCbvSizeBytes;

  // Constants for MSC descriptor heap sizes.
  static constexpr size_t kResourceHeapSlotsPerTable = 1025 + 2;
  static constexpr size_t kSamplerHeapSlotsPerTable = 257 + 2;
  static constexpr size_t kCbvHeapSlotsPerTable = 5 + 2;
  static constexpr size_t kTopLevelABSlotsPerTable = 32;
  static constexpr size_t kTopLevelABBytesPerTable =
      kTopLevelABSlotsPerTable * sizeof(uint64_t);

  // System constants population (mirrors D3D12 implementation)
  void UpdateSystemConstantValues(bool shared_memory_is_uav,
                                  bool primitive_polygonal,
                                  uint32_t line_loop_closing_index,
                                  xenos::Endian index_endian,
                                  const draw_util::ViewportInfo& viewport_info,
                                  uint32_t used_texture_mask,
                                  reg::RB_DEPTHCONTROL normalized_depth_control,
                                  uint32_t normalized_color_mask);

  // Metal device and command queue (from provider)
  MTL::Device* device_ = nullptr;
  MTL::CommandQueue* command_queue_ = nullptr;
  MTL::SharedEvent* wait_shared_event_ = nullptr;
  uint64_t wait_shared_event_value_ = 0;

  // Current command buffer and encoder
  MTL::CommandBuffer* current_command_buffer_ = nullptr;
  MTL::RenderCommandEncoder* current_render_encoder_ = nullptr;
  MTL::RenderPassDescriptor* current_render_pass_descriptor_ = nullptr;
  NS::AutoreleasePool* command_buffer_autorelease_pool_ = nullptr;

  struct EncoderResourceUsage {
    MTL::Resource* resource = nullptr;
    uint32_t usage_bits = 0;
  };
  // Tracks resources marked via useResource for the current render encoder
  // to avoid redundant driver calls across draws within the same encoder.
  std::vector<EncoderResourceUsage> render_encoder_resource_usage_;
  std::vector<MTL::Heap*> render_encoder_heap_usage_;

  // Shared memory for Xbox 360 memory access
  std::unique_ptr<MetalSharedMemory> shared_memory_;
  std::unique_ptr<MetalPrimitiveProcessor> primitive_processor_;
  bool frame_open_ = false;

  bool saw_swap_ = false;
  uint32_t last_swap_ptr_ = 0;
  uint32_t last_swap_width_ = 0;
  uint32_t last_swap_height_ = 0;
  std::unordered_map<uint32_t, bool> swap_dest_swaps_by_base_;

  // Pipeline cache (owns shaders, pipelines, shader translation components).
  std::unique_ptr<MetalPipelineCache> pipeline_cache_;

  struct DepthStencilStateKey {
    uint32_t depth_control;
    uint32_t stencil_ref_mask_front;
    uint32_t stencil_ref_mask_back;
    uint32_t polygonal_and_backface;
    bool operator==(const DepthStencilStateKey& other) const {
      return depth_control == other.depth_control &&
             stencil_ref_mask_front == other.stencil_ref_mask_front &&
             stencil_ref_mask_back == other.stencil_ref_mask_back &&
             polygonal_and_backface == other.polygonal_and_backface;
    }
    struct Hasher {
      size_t operator()(const DepthStencilStateKey& key) const {
        size_t h = size_t(key.depth_control);
        h ^= size_t(key.stencil_ref_mask_front) << 1;
        h ^= size_t(key.stencil_ref_mask_back) << 2;
        h ^= size_t(key.polygonal_and_backface) << 3;
        return h;
      }
    };
  };

  std::unordered_map<DepthStencilStateKey, MTL::DepthStencilState*,
                     DepthStencilStateKey::Hasher>
      depth_stencil_state_cache_;

  bool mesh_shader_supported_ = false;

  // Texture cache for guest texture uploads
  std::unique_ptr<MetalTextureCache> texture_cache_;

  // Render target cache for framebuffer management
  std::unique_ptr<MetalRenderTargetCache> render_target_cache_;

  // Null resources for unbound slots
  MTL::Buffer* null_buffer_ = nullptr;
  MTL::Texture* null_texture_ = nullptr;
  MTL::SamplerState* null_sampler_ = nullptr;

  // Persistent bindless descriptor heaps.
  // Canonical texture views and samplers get stable slot indices allocated on
  // demand and freed on destruction. Non-canonical texture views may use
  // submission-lifetime slots from the same heap. The heaps are bound once per
  // encoder.
  // 1M entries (24 MiB). Metal has no API-side descriptor heap cap — the heap
  // is just an MTLBuffer. D3D12 uses 262144 but doesn't need per-swizzle
  // texture views; 4x headroom covers the swizzled-view multiplier and avoids
  // exhausting the heap before the first submission completes.
  static constexpr uint32_t kViewBindlessHeapSize = 1048576;
  static constexpr uint32_t kSamplerBindlessHeapSize = 2048;
  MTL::Buffer* view_bindless_heap_ = nullptr;
  MTL::Buffer* sampler_bindless_heap_ = nullptr;
  // Explicit-layout top-level bindings for shared memory / EDRAM use small
  // dedicated system tables rather than overlapping the bindless texture heap.
  MTL::Buffer* system_view_tables_ = nullptr;
  static constexpr uint32_t kSystemViewTableSRVSharedMemory = 0;
  static constexpr uint32_t kSystemViewTableSRVNull = 1;
  static constexpr uint32_t kSystemViewTableUAVNullStart = 2;
  static constexpr uint32_t kSystemViewTableUAVSharedMemoryStart = 4;
  static constexpr uint32_t kSystemViewTableEntryCount = 6;

  // Simple bump allocator with free list for persistent heap slots.
  uint32_t view_bindless_heap_next_ = 0;
  std::vector<uint32_t> view_bindless_heap_free_;
  bool view_bindless_heap_exhausted_logged_ = false;
  uint32_t sampler_bindless_heap_next_ = 0;
  std::vector<uint32_t> sampler_bindless_heap_free_;
  bool sampler_bindless_heap_exhausted_logged_ = false;
  struct RetiredBindlessDescriptor {
    uint32_t index;
    uint64_t submission_id;
  };
  std::deque<RetiredBindlessDescriptor> retired_view_bindless_indices_;
  std::deque<RetiredBindlessDescriptor> retired_sampler_bindless_indices_;

  MTL::Buffer* tessellator_tables_buffer_ = nullptr;

  // System constants - matches DxbcShaderTranslator::SystemConstants layout
  DxbcShaderTranslator::SystemConstants system_constants_;

  // Fixed-function dynamic state cached per render encoder.
  MTL::RenderPipelineState* current_render_pipeline_state_ = nullptr;
  float ff_blend_factor_[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  bool ff_blend_factor_valid_ = false;
  bool rasterizer_state_valid_ = false;
  MTL::CullMode current_cull_mode_ = MTL::CullModeNone;
  MTL::Winding current_front_facing_winding_ = MTL::WindingCounterClockwise;
  MTL::TriangleFillMode current_triangle_fill_mode_ = MTL::TriangleFillModeFill;
  float current_depth_bias_values_[3] = {0.0f, 0.0f, 0.0f};
  MTL::DepthClipMode current_depth_clip_mode_ = MTL::DepthClipModeClip;
  MTL::DepthStencilState* current_depth_stencil_state_ = nullptr;
  bool stencil_reference_valid_ = false;
  uint32_t current_stencil_reference_ = 0;
  bool viewport_dirty_ = true;
  MTL::Viewport cached_viewport_ = {};
  bool scissor_dirty_ = true;
  MTL::ScissorRect cached_scissor_ = {};

  // Constant buffer dirty tracking (D3D12 pattern).
  // Each binding records the pool-allocated buffer, offset, and GPU address
  // for the most recent constant upload.  When up_to_date is true the data
  // from the previous allocation can be copied instead of re-gathering from
  // the register file.
  struct ConstantBufferBinding {
    MTL::Buffer* buffer = nullptr;
    NS::UInteger offset = 0;
    uint64_t gpu_address = 0;
    bool up_to_date = false;
  };
  bool cbuffer_binding_system_up_to_date_ = false;
  ConstantBufferBinding cbuffer_binding_float_vertex_;
  ConstantBufferBinding cbuffer_binding_float_pixel_;
  ConstantBufferBinding cbuffer_binding_bool_loop_;
  ConstantBufferBinding cbuffer_binding_fetch_;
  bool cbuffer_binding_descriptor_indices_vertex_up_to_date_ = false;
  bool cbuffer_binding_descriptor_indices_pixel_up_to_date_ = false;

  // Float constant usage bitmaps for the current shader pair.
  // Used to gate WriteRegister invalidation: only dirty the float CBV
  // when the written register is in the current shader's bitmap.
  // Matches D3D12's current_float_constant_map_vertex_/pixel_ at
  // d3d12_command_processor.h:829-830.
  uint64_t current_float_constant_map_vertex_[4] = {};
  uint64_t current_float_constant_map_pixel_[4] = {};
  size_t current_texture_layout_uid_vertex_ = 0;
  size_t current_texture_layout_uid_pixel_ = 0;
  size_t current_sampler_layout_uid_vertex_ = 0;
  size_t current_sampler_layout_uid_pixel_ = 0;
  std::vector<MetalTextureCache::TextureSRVKey>
      current_texture_srv_keys_vertex_;
  std::vector<MetalTextureCache::TextureSRVKey> current_texture_srv_keys_pixel_;
  std::vector<MetalTextureCache::SamplerParameters> current_samplers_vertex_;
  std::vector<MetalTextureCache::SamplerParameters> current_samplers_pixel_;
  std::vector<uint32_t> current_texture_bindless_indices_vertex_;
  std::vector<uint32_t> current_texture_bindless_indices_pixel_;
  std::vector<uint32_t> current_sampler_bindless_indices_vertex_;
  std::vector<uint32_t> current_sampler_bindless_indices_pixel_;
  bool current_bindless_table_valid_ = false;
  MTL::Buffer* current_bindless_top_level_buffer_ = nullptr;
  NS::UInteger current_bindless_top_level_offset_ = 0;
  uint64_t current_bindless_top_level_gpu_address_ = 0;
  MTL::Buffer* current_bindless_cbv_buffer_ = nullptr;
  NS::UInteger current_bindless_cbv_offset_ = 0;
  uint64_t current_bindless_cbv_gpu_address_ = 0;
  uint64_t current_bindless_vs_uniforms_gpu_ = 0;
  uint64_t current_bindless_ps_uniforms_gpu_ = 0;
  bool current_bindless_shared_memory_is_uav_ = false;
  bool current_bindless_uses_mesh_stages_ = false;

  // Pool for per-draw constant buffer allocations (replaces the ring's
  // uniforms_buffer_ for constant data).
  std::unique_ptr<MetalUploadBufferPool> constant_buffer_pool_;
  // Track which heap buffer binds have been set on the current encoder.
  bool heap_binds_set_on_encoder_ = false;

  std::atomic<uint64_t> completed_command_buffers_{0};
  std::atomic<uint32_t> pending_completion_handlers_{0};
  uint64_t submission_current_ = 0;
  uint64_t submission_completed_processed_ = 0;

  bool submission_has_draws_ = false;
  bool copy_resolve_writes_pending_ = false;

  ResolveOrderingPolicy resolve_ordering_policy_ =
      ResolveOrderingPolicy::kSubmissionBoundary;

  // Memexport tracking for shared memory invalidation.
  std::vector<draw_util::MemExportRange> memexport_ranges_;

  bool gamma_ramp_256_entry_table_up_to_date_ = false;
  bool gamma_ramp_pwl_up_to_date_ = false;

  // Trace-only resolve protection (see TraceResolveGuard above).
  TraceResolveGuard trace_resolve_guard_;
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_COMMAND_PROCESSOR_H_
