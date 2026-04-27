/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/ui/apple_ui_flags.h"

DEFINE_bool(
    ui_theme_tokens, false,
    "Enable Apple-native shared UI theme tokens for launcher and overlay UI.",
    "UI") DEFINE_bool(ios_async_import_ui, false,
                      "Enable asynchronous iOS game import flow with "
                      "progress-driven UI state.",
                      "UI")
    DEFINE_bool(
        ui_controller_navigation, true,
        "Enable Apple-native controller-first UI navigation model and focus "
        "handling.",
        "UI") DEFINE_string(macos_ui_backend, "qt",
                            "Select macOS UI backend. Use: [qt, native] "
                            "(default keeps current Qt "
                            "behavior).",
                            "UI")
