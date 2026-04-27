/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/ui/windowed_app_context_qt.h"

#include <QApplication>
#include <QMetaObject>
#include <QTimer>

#if XE_PLATFORM_WIN32
#include "xenia/base/platform_win.h"
#endif

namespace xe {
namespace ui {

QtWindowedAppContext::QtWindowedAppContext(QApplication* app)
    : app_(app)
#if XE_PLATFORM_WIN32
      ,
      hinstance_(GetModuleHandle(nullptr))
#endif
{
  pending_functions_timer_ = new QTimer();
  pending_functions_timer_->setSingleShot(true);
  QObject::connect(pending_functions_timer_, &QTimer::timeout,
                   [this]() { ExecutePendingFunctionsFromTimer(); });
}

QtWindowedAppContext::~QtWindowedAppContext() {
  shutdown_.store(true, std::memory_order_release);
  if (pending_functions_timer_) {
    pending_functions_timer_->stop();
    delete pending_functions_timer_;
    pending_functions_timer_ = nullptr;
  }
}

void QtWindowedAppContext::NotifyUILoopOfPendingFunctions() {
  if (shutdown_.load(std::memory_order_acquire)) {
    return;
  }
  // Use QMetaObject::invokeMethod on the timer to safely call from any thread
  QMetaObject::invokeMethod(
      pending_functions_timer_, [this]() { StartTimerInternal(); },
      Qt::QueuedConnection);
}

void QtWindowedAppContext::StartTimerInternal() {
  if (shutdown_.load(std::memory_order_acquire) || !pending_functions_timer_) {
    return;
  }
  if (!pending_functions_timer_->isActive()) {
    pending_functions_timer_->start(0);
  }
}

void QtWindowedAppContext::PlatformQuitFromUIThread() {
  if (app_) {
    app_->quit();
  }
}

void QtWindowedAppContext::RunMainQtLoop() {
  if (app_) {
    app_->exec();
  }
}

void QtWindowedAppContext::ExecutePendingFunctionsFromTimer() {
  ExecutePendingFunctionsFromUIThread();
}

}  // namespace ui
}  // namespace xe
