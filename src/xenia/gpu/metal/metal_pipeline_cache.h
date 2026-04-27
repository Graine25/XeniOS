/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_PIPELINE_CACHE_H_
#define XENIA_GPU_METAL_METAL_PIPELINE_CACHE_H_

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "xenia/base/string_buffer.h"
#include "xenia/gpu/dxbc_shader_translator.h"
#include "xenia/gpu/metal/dxbc_to_dxil_converter.h"
#include "xenia/gpu/metal/metal_geometry_shader.h"
#include "xenia/gpu/metal/metal_shader.h"
#include "xenia/gpu/metal/metal_shader_converter.h"
#include "xenia/gpu/primitive_processor.h"
#include "xenia/gpu/register_file.h"
#include "xenia/gpu/registers.h"
// clang-format off
// Must come after Metal.hpp (included transitively via metal_shader.h)
#include "third_party/metal-shader-converter/include/metal_irconverter_runtime.h"
// clang-format on
#include "xenia/ui/metal/metal_api.h"

namespace xe {
namespace gpu {
namespace metal {

// Attachment format snapshot used to decouple pipeline creation from render
// target cache / render pass descriptor state.
struct PipelineAttachmentFormats {
  MTL::PixelFormat color_formats[4];
  MTL::PixelFormat depth_format;
  MTL::PixelFormat stencil_format;
  uint32_t sample_count;
};

// Fixed-function rendering state derived from registers that is common to all
// pipeline paths (standard, geometry emulation, tessellation emulation).
// Computed once per draw and passed into each GetOrCreate* method so the
// duplicated register reads are eliminated.
struct PipelineRenderingKey {
  uint32_t normalized_color_mask;
  uint32_t alpha_to_mask_enable;
  uint32_t blendcontrol[4];
};

// Derive the shared PipelineRenderingKey from the current register state and
// the pixel translation (which provides writes_color_targets).
PipelineRenderingKey ResolvePipelineRenderingKey(
    const RegisterFile& regs,
    const MetalShader::MetalTranslation* pixel_translation,
    bool use_fallback_pixel_shader);

class MetalPipelineCache {
 public:
  MetalPipelineCache(MTL::Device* device, const RegisterFile& register_file);
  ~MetalPipelineCache();

  // Non-copyable.
  MetalPipelineCache(const MetalPipelineCache&) = delete;
  MetalPipelineCache& operator=(const MetalPipelineCache&) = delete;

  // Initialize shader translation components.  Must be called after
  // construction; returns false on failure.
  bool InitializeShaderTranslation(bool gamma_render_target_as_unorm8,
                                   bool msaa_2x_supported,
                                   uint32_t draw_resolution_scale_x,
                                   uint32_t draw_resolution_scale_y);

  // Shader storage (disk cache) lifecycle.
  void InitializeShaderStorage(const std::filesystem::path& cache_root,
                               uint32_t title_id, bool blocking);
  void ShutdownShaderStorage();

  // Load or retrieve a cached shader from the ucode hash.
  Shader* LoadShader(xenos::ShaderType shader_type, uint32_t guest_address,
                     const uint32_t* host_address, uint32_t dword_count);

  // Per-draw shader modification selection (mirrors D3D12 PipelineCache).
  DxbcShaderTranslator::Modification GetCurrentVertexShaderModification(
      const Shader& shader,
      Shader::HostVertexShaderType host_vertex_shader_type,
      uint32_t interpolator_mask) const;
  DxbcShaderTranslator::Modification GetCurrentPixelShaderModification(
      const Shader& shader, uint32_t interpolator_mask, uint32_t param_gen_pos,
      reg::RB_DEPTHCONTROL normalized_depth_control) const;

  // Handle for a standard render pipeline.  The state pointer is set
  // atomically so background compilation threads can publish results that
  // the render thread picks up via acquire loads.
  struct PipelineHandle {
    std::atomic<MTL::RenderPipelineState*> state{nullptr};
    // Data needed by background thread to create the pipeline.
    // Cleared after creation.
    MetalShader::MetalTranslation* pending_vertex_translation{nullptr};
    MetalShader::MetalTranslation* pending_pixel_translation{nullptr};
    PipelineAttachmentFormats pending_formats{};
    uint32_t pending_normalized_color_mask{0};
    uint32_t pending_blendcontrol[4]{};
    bool pending_alpha_to_mask{false};
    uint8_t priority{0};
  };

  // Pipeline state creation -- all accept pre-resolved attachment formats and
  // the shared rendering key derived by ResolvePipelineRenderingKey().
  PipelineHandle* GetOrCreatePipelineState(
      MetalShader::MetalTranslation* vertex_translation,
      MetalShader::MetalTranslation* pixel_translation,
      const PipelineAttachmentFormats& attachment_formats,
      const PipelineRenderingKey& rendering_key);

  // Returns true while background pipeline creation threads are busy.
  bool IsCreatingPipelines();

  struct GeometryVertexStageState {
    MTL::Library* library = nullptr;
    MTL::Library* stage_in_library = nullptr;
    std::string function_name;
    uint32_t vertex_output_size_in_bytes = 0;
  };

  struct GeometryShaderStageState {
    MTL::Library* library = nullptr;
    std::string function_name;
    uint32_t max_input_primitives_per_mesh_threadgroup = 0;
    std::vector<MetalShaderFunctionConstant> function_constants;
  };

  struct GeometryPipelineState {
    MTL::RenderPipelineState* pipeline = nullptr;
    uint32_t gs_vertex_size_in_bytes = 0;
    uint32_t gs_max_input_primitives_per_mesh_threadgroup = 0;
  };

  struct TessellationVertexStageState {
    MTL::Library* library = nullptr;
    MTL::Library* stage_in_library = nullptr;
    std::string function_name;
    uint32_t vertex_output_size_in_bytes = 0;
  };

  struct TessellationHullStageState {
    MTL::Library* library = nullptr;
    std::string function_name;
    MetalShaderReflectionInfo reflection;
  };

  struct TessellationDomainStageState {
    MTL::Library* library = nullptr;
    std::string function_name;
    MetalShaderReflectionInfo reflection;
  };

  struct TessellationPipelineState {
    MTL::RenderPipelineState* pipeline = nullptr;
    IRRuntimeTessellationPipelineConfig config = {};
    IRRuntimePrimitiveType primitive = IRRuntimePrimitiveTypeTriangle;
  };

  GeometryPipelineState* GetOrCreateGeometryPipelineState(
      MetalShader::MetalTranslation* vertex_translation,
      MetalShader::MetalTranslation* pixel_translation,
      GeometryShaderKey geometry_shader_key,
      const PipelineAttachmentFormats& attachment_formats,
      const PipelineRenderingKey& rendering_key);

  TessellationPipelineState* GetOrCreateTessellationPipelineState(
      MetalShader::MetalTranslation* domain_translation,
      MetalShader::MetalTranslation* pixel_translation,
      const PrimitiveProcessor::ProcessingResult& primitive_processing_result,
      const PipelineAttachmentFormats& attachment_formats,
      const PipelineRenderingKey& rendering_key);

  // Ensure the depth-only pixel shader is compiled and ready.
  bool EnsureDepthOnlyPixelShader();

  // Accessors for translation components (used by IssueDraw for
  // per-shader translation).
  DxbcShaderTranslator* shader_translator() { return shader_translator_.get(); }
  DxbcToDxilConverter* dxbc_to_dxil_converter() {
    return dxbc_to_dxil_converter_.get();
  }
  MetalShaderConverter* metal_shader_converter() {
    return metal_shader_converter_.get();
  }

  // Ucode disassembly scratch buffer (shared with command processor).
  StringBuffer& ucode_disasm_buffer() { return ucode_disasm_buffer_; }

  // Serialize pipeline binary archive to disk.
  void SerializePipelineBinaryArchive();

  struct PipelineDiskCacheVertexAttribute {
    uint32_t attribute_index;
    uint32_t format;
    uint32_t offset;
    uint32_t buffer_index;
  };

  struct PipelineDiskCacheVertexLayout {
    uint32_t buffer_index;
    uint32_t stride;
    uint32_t step_function;
    uint32_t step_rate;
  };

  struct PipelineDiskCacheEntry {
    uint64_t pipeline_key = 0;
    uint64_t vertex_shader_cache_key = 0;
    uint64_t pixel_shader_cache_key = 0;
    uint32_t sample_count = 1;
    uint32_t depth_format = 0;
    uint32_t stencil_format = 0;
    uint32_t color_formats[4] = {};
    uint32_t normalized_color_mask = 0;
    uint32_t alpha_to_mask_enable = 0;
    uint32_t blendcontrol[4] = {};
    std::vector<PipelineDiskCacheVertexAttribute> vertex_attributes;
    std::vector<PipelineDiskCacheVertexLayout> vertex_layouts;
  };

 private:
  bool InitializeShaderStorageInternal(const std::filesystem::path& cache_root,
                                       uint32_t title_id, bool blocking);
  std::string GetShaderStorageDeviceTag() const;
  std::string GetShaderStorageAbiTag() const;
  bool LoadPipelineDiskCache(const std::filesystem::path& path,
                             std::vector<PipelineDiskCacheEntry>* entries);
  bool AppendPipelineDiskCacheEntry(const PipelineDiskCacheEntry& entry);
  bool InitializePipelineBinaryArchive(
      const std::filesystem::path& archive_path);
  void PrewarmPipelineBinaryArchive(
      const std::vector<PipelineDiskCacheEntry>& entries);

  MTL::Device* device_;
  const RegisterFile& register_file_;

  // MSC shader translation components.
  std::unique_ptr<DxbcShaderTranslator> shader_translator_;
  std::unique_ptr<DxbcToDxilConverter> dxbc_to_dxil_converter_;
  std::unique_ptr<MetalShaderConverter> metal_shader_converter_;

  StringBuffer ucode_disasm_buffer_;

  // MSC shader cache (keyed by ucode hash).
  std::unordered_map<uint64_t, std::unique_ptr<MetalShader>> shader_cache_;

  // MSC pipeline caches (keyed by shader combination).
  std::unordered_map<uint64_t, std::unique_ptr<PipelineHandle>> pipeline_cache_;
  std::unordered_map<uint64_t, GeometryPipelineState> geometry_pipeline_cache_;
  std::unordered_map<MetalShader::MetalTranslation*, GeometryVertexStageState>
      geometry_vertex_stage_cache_;
  std::unordered_map<GeometryShaderKey, GeometryShaderStageState,
                     GeometryShaderKey::Hasher>
      geometry_shader_stage_cache_;
  std::unordered_map<uint64_t, TessellationVertexStageState>
      tessellation_vertex_stage_cache_;
  std::unordered_map<uint64_t, TessellationHullStageState>
      tessellation_hull_stage_cache_;
  std::unordered_map<uint64_t, TessellationDomainStageState>
      tessellation_domain_stage_cache_;
  std::unordered_map<uint64_t, TessellationPipelineState>
      tessellation_pipeline_cache_;

  MTL::Library* depth_only_pixel_library_ = nullptr;
  std::string depth_only_pixel_function_name_;

  // Shader storage paths.
  std::filesystem::path shader_storage_root_;
  std::filesystem::path shader_storage_local_root_;
  std::filesystem::path shader_storage_title_root_;
  std::filesystem::path metallib_cache_dir_;
  std::filesystem::path pipeline_disk_cache_path_;
  std::filesystem::path pipeline_binary_archive_path_;
  std::unordered_set<uint64_t> pipeline_disk_cache_keys_;
  std::vector<PipelineDiskCacheEntry> pipeline_disk_cache_entries_;
  FILE* pipeline_disk_cache_file_ = nullptr;
  MTL::BinaryArchive* pipeline_binary_archive_ = nullptr;
  bool pipeline_binary_archive_dirty_ = false;
  std::mutex pipeline_binary_archive_mutex_;

  // Async pipeline compilation thread pool.
  void CreationThread(size_t thread_index);
  MTL::RenderPipelineState* CreatePipelineFromHandle(
      const PipelineHandle* handle);

  std::vector<std::thread> creation_threads_;
  struct PipelineCreationRequest {
    PipelineHandle* handle{nullptr};
  };
  struct PipelineCreationPriorityCompare {
    bool operator()(const PipelineCreationRequest& a,
                    const PipelineCreationRequest& b) const {
      return a.handle->priority < b.handle->priority;
    }
  };
  std::priority_queue<PipelineCreationRequest,
                      std::vector<PipelineCreationRequest>,
                      PipelineCreationPriorityCompare>
      creation_queue_;
  std::mutex creation_request_lock_;
  std::condition_variable creation_request_cond_;
  std::atomic<size_t> creation_threads_busy_{0};
  bool creation_threads_shutdown_{false};
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_PIPELINE_CACHE_H_
