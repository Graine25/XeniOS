/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/ui/apple_theme_tokens.h"

namespace xe {
namespace ui {
namespace apple {

namespace {

constexpr ThemeTokens kThemeDark = {
    ThemeColorTokens{
        /* bg_primary */ {0x09, 0x09, 0x0B, 0xFF},
        /* bg_surface */ {0x18, 0x18, 0x1B, 0xFF},
        /* bg_surface_2 */ {0x27, 0x27, 0x2A, 0xFF},
        /* bg_surface_3 */ {0x3F, 0x3F, 0x46, 0xFF},
        /* text_primary */ {0xFA, 0xFA, 0xFA, 0xFF},
        /* text_secondary */ {0xA1, 0xA1, 0xAA, 0xFF},
        /* text_muted */ {0x71, 0x71, 0x7A, 0xFF},
        /* accent */ {0x34, 0xD3, 0x99, 0xFF},
        /* accent_hover */ {0x6E, 0xE7, 0xB7, 0xFF},
        /* accent_fg */ {0x09, 0x09, 0x0B, 0xFF},
        // Website uses oklch(1 0 0 / 0.06) and oklch(1 0 0 / 0.1).
        /* border */ {0xFF, 0xFF, 0xFF, 15},
        /* border_hover */ {0xFF, 0xFF, 0xFF, 26},
        /* status_playable */ {0x34, 0xD3, 0x99, 0xFF},
        /* status_ingame */ {0x60, 0xA5, 0xFA, 0xFF},
        /* status_intro */ {0xFB, 0xBF, 0x24, 0xFF},
        /* status_loads */ {0xFB, 0x92, 0x3C, 0xFF},
        /* status_nothing */ {0xF8, 0x71, 0x71, 0xFF},
        /* overlay */ {0x00, 0x00, 0x00, 217},
        /* overlay_light */ {0x00, 0x00, 0x00, 148},
    },
    ThemeTypographyTokens{
        /* font_sans */ "Inter",
        /* font_mono */ "JetBrains Mono",
    },
    ThemeRadiusTokens{
        /* md */ 8.0f,
        /* lg */ 12.0f,
        /* xl */ 16.0f,
    },
};

constexpr ThemeTokens kThemeLight = {
    ThemeColorTokens{
        /* bg_primary */ {0xFF, 0xFF, 0xFF, 0xFF},
        /* bg_surface */ {0xF4, 0xF4, 0xF5, 0xFF},
        /* bg_surface_2 */ {0xE4, 0xE4, 0xE7, 0xFF},
        /* bg_surface_3 */ {0xD4, 0xD4, 0xD8, 0xFF},
        /* text_primary */ {0x09, 0x09, 0x0B, 0xFF},
        /* text_secondary */ {0x52, 0x52, 0x5B, 0xFF},
        /* text_muted */ {0xA1, 0xA1, 0xAA, 0xFF},
        /* accent */ {0x10, 0xB9, 0x81, 0xFF},
        /* accent_hover */ {0x05, 0x96, 0x69, 0xFF},
        /* accent_fg */ {0xFF, 0xFF, 0xFF, 0xFF},
        // Website uses oklch(0 0 0 / 0.08) and oklch(0 0 0 / 0.12).
        /* border */ {0x00, 0x00, 0x00, 20},
        /* border_hover */ {0x00, 0x00, 0x00, 31},
        /* status_playable */ {0x34, 0xD3, 0x99, 0xFF},
        /* status_ingame */ {0x60, 0xA5, 0xFA, 0xFF},
        /* status_intro */ {0xFB, 0xBF, 0x24, 0xFF},
        /* status_loads */ {0xFB, 0x92, 0x3C, 0xFF},
        /* status_nothing */ {0xF8, 0x71, 0x71, 0xFF},
        /* overlay */ {0x00, 0x00, 0x00, 217},
        /* overlay_light */ {0x00, 0x00, 0x00, 148},
    },
    ThemeTypographyTokens{
        /* font_sans */ "Inter",
        /* font_mono */ "JetBrains Mono",
    },
    ThemeRadiusTokens{
        /* md */ 8.0f,
        /* lg */ 12.0f,
        /* xl */ 16.0f,
    },
};

}  // namespace

const ThemeTokens& GetThemeTokens(ThemeVariant variant) {
  return variant == ThemeVariant::kLight ? kThemeLight : kThemeDark;
}

const ThemeTokens& GetDarkThemeTokens() { return kThemeDark; }

const ThemeTokens& GetLightThemeTokens() { return kThemeLight; }

}  // namespace apple
}  // namespace ui
}  // namespace xe
