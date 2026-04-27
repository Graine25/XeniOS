/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/trace_dump.h"

#include "third_party/stb/stb_image_write.h"
#include "xenia/base/cvar.h"
#include "xenia/base/filesystem.h"
#include "xenia/base/logging.h"
#include "xenia/base/profiling.h"
#include "xenia/base/string.h"
#include "xenia/base/threading.h"
#include "xenia/config.h"
#include "xenia/gpu/command_processor.h"
#include "xenia/gpu/graphics_system.h"
#include "xenia/memory.h"
#include "xenia/storage_flags.h"
#include "xenia/ui/file_picker.h"
#include "xenia/ui/presenter.h"
#include "xenia/ui/window.h"
#include "xenia/xbox.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#undef _CRT_SECURE_NO_WARNINGS
#undef _CRT_NONSTDC_NO_DEPRECATE
#include "third_party/stb/stb_image_write.h"

DEFINE_path(target_trace_file, "", "Specifies the trace file to load.", "GPU");
DEFINE_path(trace_dump_path, "", "Output path for dumped files.", "GPU");

namespace xe {
namespace gpu {

using namespace xe::gpu::xenos;

namespace {

std::filesystem::path ResolveStorageRoot() {
  std::filesystem::path storage_root = cvars::storage_root;
  if (storage_root.empty()) {
    storage_root = xe::filesystem::GetExecutableFolder();
    if (!cvars::portable &&
        !std::filesystem::exists(storage_root / "portable.txt")) {
      storage_root = xe::filesystem::GetUserFolder() / "Xenia";
    }
  }
  return std::filesystem::absolute(storage_root);
}

std::filesystem::path ResolveContentRoot(
    const std::filesystem::path& storage_root) {
  std::filesystem::path content_root = cvars::content_root;
  if (content_root.empty()) {
    content_root = storage_root / "content";
  } else if (!content_root.is_absolute()) {
    content_root = storage_root / content_root;
  }
  return std::filesystem::absolute(content_root);
}

std::filesystem::path ResolveCacheRoot(
    const std::filesystem::path& storage_root) {
  std::filesystem::path cache_root = cvars::cache_root;
  if (cache_root.empty()) {
    cache_root = storage_root / "cache_host";
  } else if (!cache_root.is_absolute()) {
    cache_root = storage_root / cache_root;
  }
  return std::filesystem::absolute(cache_root);
}

}  // namespace

TraceDump::TraceDump() = default;

TraceDump::~TraceDump() = default;

int TraceDump::Main(const std::vector<std::string>& args) {
  // Grab path from the flag or unnamed argument.
  std::filesystem::path path;
  std::filesystem::path output_path;
  if (!cvars::target_trace_file.empty()) {
    // Passed as a named argument.
    // TODO(benvanik): find something better than gflags that supports
    // unicode.
    path = cvars::target_trace_file;
  } else if (args.size() >= 2) {
    // Passed as an unnamed argument.
    path = xe::to_path(args[1]);

    if (args.size() >= 3) {
      output_path = xe::to_path(args[2]);
    }
  }

  if (path.empty()) {
    XELOGE("No trace file specified");
    return 5;
  }

  // Normalize the path and make absolute.
  auto abs_path = std::filesystem::absolute(path);
  XELOGI("Loading trace file {}...", abs_path);

  if (!Setup()) {
    XELOGE("Unable to setup trace dump tool");
    return 4;
  }
  if (!Load(std::move(abs_path))) {
    XELOGE("Unable to load trace file; not found?");
    return 5;
  }

  // Root file name for outputs.
  if (output_path.empty()) {
    base_output_path_ = cvars::trace_dump_path;
    auto output_name = path.filename().replace_extension();

    base_output_path_ = base_output_path_ / output_name;
  } else {
    base_output_path_ = output_path;
  }

  // Ensure output path exists.
  xe::filesystem::CreateParentFolder(base_output_path_);

  return Run();
}

bool TraceDump::Setup() {
  storage_root_ = ResolveStorageRoot();
  content_root_ = ResolveContentRoot(storage_root_);
  cache_root_ = ResolveCacheRoot(storage_root_);

  XELOGI("Storage root: {}", storage_root_);
  XELOGI("Content root: {}", content_root_);
  XELOGI("Host cache root: {}", cache_root_);

  config::SetupConfig(storage_root_);

  // Create the emulator but don't initialize so we can setup the window.
  emulator_ =
      std::make_unique<Emulator>("", storage_root_, content_root_, cache_root_);
  X_STATUS result = emulator_->Setup(
      nullptr, nullptr, false, nullptr,
      [this]() { return CreateGraphicsSystem(); }, nullptr);
  if (XFAILED(result)) {
    XELOGE("Failed to setup emulator: {:08X}", result);
    return false;
  }
  graphics_system_ = emulator_->graphics_system();
  graphics_system_->EnsurePresenterForCapture();
  player_ = std::make_unique<TracePlayer>(graphics_system_);
  return true;
}

bool TraceDump::Load(const std::filesystem::path& trace_file_path) {
  trace_file_path_ = trace_file_path;

  if (!player_->Open(xe::path_to_utf8(trace_file_path_))) {
    XELOGE("Could not load trace file");
    return false;
  }

  if (graphics_system_) {
    uint32_t title_id = player_->header()->title_id;
    if (title_id != 0) {
      XELOGI("Trace dump cache root: {}", cache_root_);
      // Trace dumps begin rendering immediately, so prewarm storage
      // synchronously to maximize warm-cache reuse on short runs.
      graphics_system_->InitializeShaderStorage(cache_root_, title_id, true,
                                                nullptr);
    }
  }

  return true;
}

int TraceDump::Run() {
  BeginHostCapture();
  player_->SeekFrame(0);
  player_->SeekCommand(
      static_cast<int>(player_->current_frame()->commands.size() - 1));
  player_->WaitOnPlayback();
  EndHostCapture();

  // Capture.
  int result = 0;
  ui::Presenter* presenter = graphics_system_->presenter();
  ui::RawImage raw_image;
  if (presenter && presenter->CaptureGuestOutput(raw_image)) {
    // Save framebuffer png.
    auto png_path = base_output_path_.replace_extension(".png");
    auto handle = filesystem::OpenFile(png_path, "wb");
    auto callback = [](void* context, void* data, int size) {
      fwrite(data, 1, size, (FILE*)context);
    };
    stbi_write_png_to_func(callback, handle, static_cast<int>(raw_image.width),
                           static_cast<int>(raw_image.height), 4,
                           raw_image.data.data(),
                           static_cast<int>(raw_image.stride));
    fclose(handle);
  } else {
    result = 1;
  }

  player_.reset();
  emulator_.reset();
  return result;
}

}  //  namespace gpu
}  //  namespace xe
