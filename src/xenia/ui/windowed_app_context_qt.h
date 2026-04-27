/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_UI_WINDOWED_APP_CONTEXT_QT_H_
#define XENIA_UI_WINDOWED_APP_CONTEXT_QT_H_

#include <atomic>
#include <mutex>

#include "xenia/base/platform.h"
#include "xenia/ui/windowed_app_context.h"

#if XE_PLATFORM_WIN32
#include "xenia/base/platform_win.h"
#endif

class QApplication;
class QTimer;

namespace xe {
namespace ui {

class QtWindowedAppContext final : public WindowedAppContext {
 public:
  explicit QtWindowedAppContext(QApplication* app);
  ~QtWindowedAppContext();

#if XE_PLATFORM_WIN32
  HINSTANCE hinstance() const { return hinstance_; }
#endif

  void NotifyUILoopOfPendingFunctions() override;

  void PlatformQuitFromUIThread() override;

  void RunMainQtLoop();

 private:
  void ExecutePendingFunctionsFromTimer();
  void StartTimerInternal();

  QApplication* app_;
  QTimer* pending_functions_timer_ = nullptr;
  std::atomic<bool> shutdown_{false};

#if XE_PLATFORM_WIN32
  HINSTANCE hinstance_;
#endif
};

}  // namespace ui
}  // namespace xe

#endif  // XENIA_UI_WINDOWED_APP_CONTEXT_QT_H_
