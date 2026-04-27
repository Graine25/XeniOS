/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_UI_APPLE_THEME_TOKENS_H_
#define XENIA_UI_APPLE_THEME_TOKENS_H_

#include <cstdint>

namespace xe {
namespace ui {
namespace apple {

// Snapshot source:
//   /Users/admin/Documents/xenios-website/src/app/globals.css
// Keep this module as the canonical token definition for Apple-native UI.
enum class ThemeVariant : uint8_t {
  kDark = 0,
  kLight = 1,
};

struct ColorRGBA8 {
  uint8_t r;
  uint8_t g;
  uint8_t b;
  uint8_t a;

  constexpr float Rf() const { return float(r) / 255.0f; }
  constexpr float Gf() const { return float(g) / 255.0f; }
  constexpr float Bf() const { return float(b) / 255.0f; }
  constexpr float Af() const { return float(a) / 255.0f; }
};

struct ThemeColorTokens {
  ColorRGBA8 bg_primary;
  ColorRGBA8 bg_surface;
  ColorRGBA8 bg_surface_2;
  ColorRGBA8 bg_surface_3;
  ColorRGBA8 text_primary;
  ColorRGBA8 text_secondary;
  ColorRGBA8 text_muted;
  ColorRGBA8 accent;
  ColorRGBA8 accent_hover;
  ColorRGBA8 accent_fg;
  ColorRGBA8 border;
  ColorRGBA8 border_hover;
  ColorRGBA8 status_playable;
  ColorRGBA8 status_ingame;
  ColorRGBA8 status_intro;
  ColorRGBA8 status_loads;
  ColorRGBA8 status_nothing;
  ColorRGBA8 overlay;
  ColorRGBA8 overlay_light;
};

struct ThemeTypographyTokens {
  const char* font_sans;
  const char* font_mono;
};

struct ThemeRadiusTokens {
  float md;
  float lg;
  float xl;
};

struct ThemeTokens {
  ThemeColorTokens colors;
  ThemeTypographyTokens typography;
  ThemeRadiusTokens radius;
};

const ThemeTokens& GetThemeTokens(ThemeVariant variant);
const ThemeTokens& GetDarkThemeTokens();
const ThemeTokens& GetLightThemeTokens();

}  // namespace apple
}  // namespace ui
}  // namespace xe

#endif  // XENIA_UI_APPLE_THEME_TOKENS_H_
