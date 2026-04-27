/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_UI_APPLE_UI_FLAGS_H_
#define XENIA_UI_APPLE_UI_FLAGS_H_

#include "xenia/base/cvar.h"

// Rollout gates for Apple-native UI unification work.
// Defaults intentionally preserve current behavior.
DECLARE_bool(ui_theme_tokens);
DECLARE_bool(ios_async_import_ui);
DECLARE_bool(ui_controller_navigation);
DECLARE_string(macos_ui_backend);

#endif  // XENIA_UI_APPLE_UI_FLAGS_H_
