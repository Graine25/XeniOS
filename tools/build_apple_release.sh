#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: tools/build_apple_release.sh [options]

Builds and packages Apple release artifacts:
- macOS arm64 DMG
- macOS x86_64 DMG
- iOS arm64 ad-hoc-signed IPA

Options:
  --out DIR            Output directory (default: scratch/artifacts)
  --config NAME        checked|debug|release|valgrind (default: release)
  --version VERSION    Marketing version override (defaults to latest vX.Y.Z tag, then 1.0.1)
  --build-number NUM   Build number override (defaults to git commit count)
  --channel NAME       release|preview (default: release)
  --stage NAME         alpha|beta|rc|stable (default: stable)
  --attestation-key VALUE
                       HMAC key used to attest official CI builds
  --attestation-key-file FILE
                       File containing the HMAC key used for attestation
  --stamp-bundle PATH  Stamp version/attestation metadata into an existing .app bundle and exit
  --platform NAME      Platform for --stamp-bundle (ios|macos)
  --ios-min VERSION    iOS minimum version (default: 16.0)
  --macos-min VERSION  macOS minimum version (default: 15.0)
  --mac-sign IDENTITY  macOS codesign identity (default: ad-hoc '-')
  --attestation-key-id ID
                       Optional key id embedded in the attestation payload
  --print-metadata     Print resolved version/build/channel/attestation data and exit
  --skip-ios
  --skip-macos-arm64
  --skip-macos-x86_64
  -h, --help

Notes:
- iOS packaging creates an ad-hoc-signed .ipa suitable for re-signing.
- This script expects Xcode command line tools (xcodebuild, codesign, hdiutil).
- Marketing version defaults to the latest reachable vX.Y.Z git tag.
- If no matching tag is available, the marketing version falls back to 1.0.1.
- Public release stage defaults to stable.
- Official build numbers default to git rev-list --count HEAD.
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

need_bin() {
  command -v "$1" >/dev/null 2>&1 || die "missing required tool: $1"
}

need_file_exec() {
  [ -x "$1" ] || die "missing required executable: $1"
}

cap_config() {
  # checked -> Checked, debug -> Debug, release -> Release, valgrind -> Valgrind
  local s
  s="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  if [ -z "$s" ]; then
    return 0
  fi
  local first rest
  first="$(printf '%s' "$s" | cut -c1 | tr '[:lower:]' '[:upper:]')"
  rest="$(printf '%s' "$s" | cut -c2-)"
  printf '%s%s' "$first" "$rest"
}

repo_root() {
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd
}

trim_string() {
  printf '%s' "${1:-}" | tr '\r\n' ' ' | sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//'
}

default_marketing_version() {
  printf '%s' "1.0.1"
}

validate_marketing_version() {
  [[ "${1:-}" =~ ^[0-9]+(\.[0-9]+){1,2}$ ]]
}

validate_build_number() {
  [[ "${1:-}" =~ ^[0-9]+(\.[0-9]+){0,2}$ ]]
}

validate_release_stage() {
  case "${1:-}" in
    alpha|beta|rc|stable)
      return 0
      ;;
  esac
  return 1
}

read_latest_marketing_version_from_git() {
  local root="$1"
  command -v git >/dev/null 2>&1 || return 1
  git -C "$root" rev-parse --is-inside-work-tree >/dev/null 2>&1 || return 1

  local tag=""
  local patterns=(
    "v[0-9]*.[0-9]*.[0-9]*"
    "[0-9]*.[0-9]*.[0-9]*"
    "v[0-9]*.[0-9]*"
    "[0-9]*.[0-9]*"
  )
  local pattern=""
  for pattern in "${patterns[@]}"; do
    tag="$(git -C "$root" describe --tags --abbrev=0 --match "$pattern" 2>/dev/null || true)"
    if [ -n "$tag" ]; then
      break
    fi
  done
  [ -n "$tag" ] || return 1

  tag="$(trim_string "$tag")"
  tag="${tag#v}"
  validate_marketing_version "$tag" || return 1
  printf '%s' "$tag"
}

resolve_release_version() {
  local root="$1"
  local version="${2:-}"
  if [ -n "$version" ]; then
    validate_marketing_version "$version" || die "invalid marketing version: $version"
    printf '%s' "$version"
    return 0
  fi

  if version="$(read_latest_marketing_version_from_git "$root")"; then
    printf '%s' "$version"
    return 0
  fi

  default_marketing_version
}

resolve_release_stage() {
  local stage="${1:-}"
  if [ -n "$stage" ]; then
    stage="$(trim_string "$stage" | tr '[:upper:]' '[:lower:]')"
    validate_release_stage "$stage" || die "invalid release stage: $stage"
    printf '%s' "$stage"
    return 0
  fi

  printf '%s' "stable"
}

resolve_release_build_number() {
  local build_number="${1:-}"
  if [ -n "$build_number" ]; then
    validate_build_number "$build_number" || die "invalid build number: $build_number"
    printf '%s' "$build_number"
    return 0
  fi

  build_number="$(git rev-list --count HEAD)"
  local run_attempt="${GITHUB_RUN_ATTEMPT:-1}"
  if [ -n "$run_attempt" ] && [ "$run_attempt" != "1" ]; then
    build_number="${build_number}.${run_attempt}"
  fi
  validate_build_number "$build_number" || die "invalid derived build number: $build_number"
  printf '%s' "$build_number"
}

find_first_app() {
  local dir="$1"
  local app=""
  app="$(find "$dir" -maxdepth 1 -type d -name "*.app" | LC_ALL=C sort | head -n 1 || true)"
  [ -n "$app" ] || return 1
  printf '%s' "$app"
}

bundle_executable_path() {
  local app_bundle="$1"
  local plist
  plist="$(bundle_plist_path "$app_bundle")" || return 1
  local executable_name
  executable_name="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleExecutable' "$plist" 2>/dev/null || true)"
  [ -n "$executable_name" ] || return 1
  if [ -d "$app_bundle/Contents/MacOS" ]; then
    printf '%s/%s' "$app_bundle/Contents/MacOS" "$executable_name"
    return 0
  fi
  printf '%s/%s' "$app_bundle" "$executable_name"
}

resolve_macdeployqt() {
  if [ -n "${MACDEPLOYQT:-}" ] && [ -x "${MACDEPLOYQT}" ]; then
    echo "${MACDEPLOYQT}"
    return 0
  fi
  if command -v macdeployqt >/dev/null 2>&1; then
    command -v macdeployqt
    return 0
  fi
  local candidates=(
    "/opt/homebrew/opt/qt/bin/macdeployqt"
    "/usr/local/opt/qt/bin/macdeployqt"
  )
  for c in "${candidates[@]}"; do
    if [ -x "$c" ]; then
      echo "$c"
      return 0
    fi
  done
  return 1
}

normalize_version() {
  local v="${1:-0}"
  awk -v v="$v" '
    BEGIN {
      n = split(v, a, ".")
      for (i = 1; i <= 3; ++i) {
        printf("%d", (i <= n ? a[i] + 0 : 0))
        if (i < 3) printf(".")
      }
    }'
}

version_eq() {
  [ "$(normalize_version "$1")" = "$(normalize_version "$2")" ]
}

version_le() {
  local a b
  a="$(normalize_version "$1")"
  b="$(normalize_version "$2")"
  awk -v a="$a" -v b="$b" '
    BEGIN {
      split(a, av, ".")
      split(b, bv, ".")
      for (i = 1; i <= 3; ++i) {
        ai = av[i] + 0
        bi = bv[i] + 0
        if (ai < bi) { exit 0 }
        if (ai > bi) { exit 1 }
      }
      exit 0
    }'
}

sanitize_attestation_value() {
  printf '%s' "${1:-}" | tr '\r\n' ' ' | sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//'
}

sanitize_build_fragment() {
  printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9.-]+/-/g; s/^-+//; s/-+$//'
}

make_build_id() {
  local platform="$1"
  local channel="$2"
  local version="$3"
  local build_number="$4"
  local stage="${5:-}"
  local p c v b s
  p="$(sanitize_build_fragment "$platform")"
  c="$(sanitize_build_fragment "$channel")"
  v="$(sanitize_build_fragment "$version")"
  b="$(sanitize_build_fragment "$build_number")"
  if [ -z "$p" ] || [ -z "$c" ] || [ -z "$v" ] || [ -z "$b" ]; then
    return 1
  fi
  if [ -n "$stage" ] && [ "$stage" != "stable" ]; then
    s="$(sanitize_build_fragment "$stage")"
    [ -n "$s" ] || return 1
    printf '%s-%s-%s-%s-%s' "$p" "$c" "$v" "$b" "$s"
    return 0
  fi
  printf '%s-%s-%s-%s' "$p" "$c" "$v" "$b"
}

bundle_plist_path() {
  local app_bundle="$1"
  if [ -f "$app_bundle/Contents/Info.plist" ]; then
    printf '%s' "$app_bundle/Contents/Info.plist"
    return 0
  fi
  if [ -f "$app_bundle/Info.plist" ]; then
    printf '%s' "$app_bundle/Info.plist"
    return 0
  fi
  return 1
}

plist_delete_key() {
  local plist="$1"
  local key="$2"
  /usr/libexec/PlistBuddy -c "Delete :$key" "$plist" >/dev/null 2>&1 || true
}

plist_set_string() {
  local plist="$1"
  local key="$2"
  local value="$3"
  plist_delete_key "$plist" "$key"
  /usr/libexec/PlistBuddy -c "Add :$key string $value" "$plist" >/dev/null
}

plist_set_bool() {
  local plist="$1"
  local key="$2"
  local value="$3"
  plist_delete_key "$plist" "$key"
  /usr/libexec/PlistBuddy -c "Add :$key bool $value" "$plist" >/dev/null
}

plist_merge() {
  local plist="$1"
  local other_plist="$2"
  [ -f "$other_plist" ] || return 0
  /usr/libexec/PlistBuddy -c "Merge $other_plist" "$plist" >/dev/null
}

bundle_resources_path() {
  local app_bundle="$1"
  if [ -d "$app_bundle/Contents/Resources" ]; then
    printf '%s' "$app_bundle/Contents/Resources"
    return 0
  fi
  printf '%s' "$app_bundle"
}

clean_compiled_icon_assets() {
  local resources_dir="$1"
  [ -d "$resources_dir" ] || return 0
  rm -f "$resources_dir/Assets.car"
  rm -f "$resources_dir/AppIcon.icns"
  rm -f "$resources_dir"/AppIcon*.png
}

compile_bundle_icon_assets() {
  local app_bundle="$1"
  local platform="$2"
  local min_version="$3"
  local icon_catalog="$root/assets/apple/AppIcon.xcassets"

  [ -d "$icon_catalog" ] || die "missing icon catalog: $icon_catalog"

  local resources_dir plist partial_plist
  resources_dir="$(bundle_resources_path "$app_bundle")"
  plist="$(bundle_plist_path "$app_bundle")" || die "Info.plist not found in $app_bundle"
  mkdir -p "$resources_dir"
  clean_compiled_icon_assets "$resources_dir"

  partial_plist="$(mktemp "${TMPDIR:-/tmp}/xenios_icon_plist.XXXXXX")"
  rm -f "$partial_plist"

  local -a actool_cmd=(
    xcrun actool
    --compile "$resources_dir"
    --app-icon AppIcon
    --platform "$platform"
    --minimum-deployment-target "$min_version"
    --output-partial-info-plist "$partial_plist"
  )

  if [ "$platform" = "iphoneos" ]; then
    actool_cmd+=(--target-device iphone --target-device ipad)
    plist_delete_key "$plist" "CFBundleIcons"
    plist_delete_key "$plist" "CFBundleIcons~ipad"
  else
    plist_delete_key "$plist" "CFBundleIconFile"
    plist_delete_key "$plist" "CFBundleIconName"
  fi

  actool_cmd+=("$icon_catalog")
  "${actool_cmd[@]}" >/dev/null
  plist_merge "$plist" "$partial_plist"
  rm -f "$partial_plist"
}

stamp_bundle_version_metadata() {
  local app_bundle="$1"
  local version="$2"
  local build_number="$3"
  [ -n "$version" ] || return 0
  [ -n "$build_number" ] || return 0
  local plist
  plist="$(bundle_plist_path "$app_bundle")" || die "Info.plist not found in $app_bundle"
  plist_set_string "$plist" "CFBundleShortVersionString" "$version"
  plist_set_string "$plist" "CFBundleVersion" "$build_number"
}

clear_bundle_attestation_metadata() {
  local app_bundle="$1"
  local plist
  plist="$(bundle_plist_path "$app_bundle")" || die "Info.plist not found in $app_bundle"
  plist_delete_key "$plist" "XeniOSBuildChannel"
  plist_delete_key "$plist" "XeniOSBuildOfficial"
  plist_delete_key "$plist" "XeniOSBuildAttestationPayload"
  plist_delete_key "$plist" "XeniOSBuildAttestationSignature"
}

stamp_bundle_stage_metadata() {
  local app_bundle="$1"
  local stage="$2"
  local plist
  plist="$(bundle_plist_path "$app_bundle")" || die "Info.plist not found in $app_bundle"
  plist_delete_key "$plist" "XeniOSBuildStage"
  [ -n "$stage" ] || return 0
  plist_set_string "$plist" "XeniOSBuildStage" "$stage"
}

build_attestation_payload() {
  local platform="$1"
  local channel="$2"
  local build_id="$3"
  local version="$4"
  local build_number="$5"
  local stage="$6"
  local commit_short="$7"
  local issued_at="$8"
  local key_id="$9"
  printf '%s' \
    "xenios-build-attestation-v1;platform=$(sanitize_attestation_value "$platform");channel=$(sanitize_attestation_value "$channel");buildId=$(sanitize_attestation_value "$build_id");appVersion=$(sanitize_attestation_value "$version");buildNumber=$(sanitize_attestation_value "$build_number");stage=$(sanitize_attestation_value "$stage");commitShort=$(sanitize_attestation_value "$commit_short");issuedAt=$(sanitize_attestation_value "$issued_at");keyId=$(sanitize_attestation_value "$key_id")"
}

sign_attestation_payload() {
  local payload="$1"
  local key="$2"
  printf '%s' "$payload" | \
    openssl dgst -sha256 -mac HMAC -macopt "key:$key" -binary | \
    openssl base64 -A | tr '+/' '-_' | tr -d '='
}

stamp_bundle_attestation() {
  local app_bundle="$1"
  local platform="$2"
  local channel="$3"
  local version="$4"
  local build_number="$5"
  local stage="$6"
  local commit_short="$7"
  local issued_at="$8"
  local key_id="$9"
  local attestation_key="${10}"

  clear_bundle_attestation_metadata "$app_bundle"

  if [ -z "$attestation_key" ]; then
    return 0
  fi

  local build_id
  build_id="$(make_build_id "$platform" "$channel" "$version" "$build_number" "$stage")" || \
    die "official build attestation requires version and build number"
  local payload
  payload="$(build_attestation_payload "$platform" "$channel" "$build_id" "$version" \
    "$build_number" "$stage" "$commit_short" "$issued_at" "$key_id")"
  local signature
  signature="$(sign_attestation_payload "$payload" "$attestation_key")"

  local plist
  plist="$(bundle_plist_path "$app_bundle")" || die "Info.plist not found in $app_bundle"
  plist_set_string "$plist" "XeniOSBuildChannel" "$channel"
  plist_set_bool "$plist" "XeniOSBuildOfficial" true
  plist_set_string "$plist" "XeniOSBuildAttestationPayload" "$payload"
  plist_set_string "$plist" "XeniOSBuildAttestationSignature" "$signature"
}

dylib_macos_deployment_target() {
  local dylib="$1"
  otool -l "$dylib" | awk '
    $1 == "cmd" && $2 == "LC_BUILD_VERSION" { mode = "build"; next }
    $1 == "cmd" && $2 == "LC_VERSION_MIN_MACOSX" { mode = "legacy"; next }
    $1 == "cmd" { mode = "" }
    mode == "build" && $1 == "minos" { print $2; exit }
    mode == "legacy" && $1 == "version" { print $2; exit }
  '
}

deploy_qt_if_missing() {
  local app_bundle="$1"
  local qt_core="$app_bundle/Contents/Frameworks/QtCore.framework/Versions/A/QtCore"
  local qcocoa="$app_bundle/Contents/PlugIns/platforms/libqcocoa.dylib"

  if [ -f "$qt_core" ] && [ -f "$qcocoa" ]; then
    echo "Qt already bundled by post-build step; skipping extra macdeployqt pass."
    return 0
  fi

  local macdeployqt=""
  if macdeployqt="$(resolve_macdeployqt)"; then
    "$macdeployqt" "$app_bundle" -always-overwrite \
      -libpath=third_party/metal-shader-converter/lib \
      -libpath=third_party/DirectXShaderCompiler/build_dxilconv_macos/lib \
      -libpath=third_party/DirectXShaderCompiler/build_dxilconv_macos_x86_64/lib \
      -libpath=/opt/homebrew/opt/sdl2/lib \
      -libpath=/usr/local/opt/sdl2/lib
  else
    echo "warning: macdeployqt not found; app may be missing Qt frameworks"
  fi
}

require_sdl2_deployment_target() {
  local app_bundle="$1"
  local required_target="$2"
  local app_frameworks="$app_bundle/Contents/Frameworks"
  local sdl_dylib="$app_frameworks/libSDL2-2.0.0.dylib"
  [ -f "$sdl_dylib" ] || die "bundled SDL2 not found: $sdl_dylib"

  local actual_target
  actual_target="$(dylib_macos_deployment_target "$sdl_dylib")"
  [ -n "$actual_target" ] || die "unable to read deployment target from $sdl_dylib"

  if ! version_le "$actual_target" "$required_target"; then
    die "SDL2 deployment target too new: got $actual_target, app target is $required_target ($sdl_dylib)"
  fi

  if version_eq "$actual_target" "$required_target"; then
    echo "SDL2 deployment target OK: $actual_target ($sdl_dylib)"
  else
    echo "SDL2 deployment target compatible: $actual_target (app target: $required_target) ($sdl_dylib)"
  fi
}

sign_macos_app() {
  local app_bundle="$1"
  local entitlements="$2"
  local identity="${3:--}"

  local main_exe=""
  main_exe="$(bundle_executable_path "$app_bundle" || true)"
  local frameworks_dir="$app_bundle/Contents/Frameworks"
  local plugins_dir="$app_bundle/Contents/PlugIns"

  if [ -f "$main_exe" ]; then
    codesign --remove-signature "$main_exe" >/dev/null 2>&1 || true
  fi
  codesign --remove-signature "$app_bundle" >/dev/null 2>&1 || true

  if [ -d "$frameworks_dir" ]; then
    find "$frameworks_dir" -maxdepth 2 -name "*.framework" -type d -print0 | \
      xargs -0 -I{} codesign --remove-signature "{}" >/dev/null 2>&1 || true
    find "$frameworks_dir" -name "*.dylib" -type f -print0 | \
      xargs -0 -I{} codesign --remove-signature "{}" >/dev/null 2>&1 || true

    find "$frameworks_dir" -maxdepth 2 -name "*.framework" -type d -print0 | \
      xargs -0 -I{} codesign --force --sign "$identity" --timestamp=none "{}"
    find "$frameworks_dir" -name "*.dylib" -type f -print0 | \
      xargs -0 -I{} codesign --force --sign "$identity" --timestamp=none "{}"
  fi

  if [ -d "$plugins_dir" ]; then
    find "$plugins_dir" -name "*.dylib" -type f -print0 | \
      xargs -0 -I{} codesign --remove-signature "{}" >/dev/null 2>&1 || true
    find "$plugins_dir" -name "*.dylib" -type f -print0 | \
      xargs -0 -I{} codesign --force --sign "$identity" --timestamp=none "{}"
  fi

  if [ -f "$main_exe" ]; then
    codesign --force --sign "$identity" --timestamp=none "$main_exe"
  fi

  codesign --force --deep --timestamp=none --sign "$identity" \
    --entitlements "$entitlements" "$app_bundle"

  # Some Homebrew dylibs can be copied with read-only mode and carry provenance
  # xattrs, making recursive xattr clearing fail with EPERM.
  if ! xattr -cr "$app_bundle" >/dev/null 2>&1; then
    chmod -R u+w "$app_bundle" >/dev/null 2>&1 || true
    if ! xattr -cr "$app_bundle" >/dev/null 2>&1; then
      echo "warning: failed to clear one or more extended attributes in $app_bundle"
    fi
  fi
}

package_macos_dmg() {
  local app_bundle="$1"
  local dmg_out="$2"
  local license_file="$3"

  rm -f "$dmg_out"
  local dmg_contents
  dmg_contents="$(mktemp -d "${TMPDIR:-/tmp}/xenia_dmg.XXXXXX")"
  mkdir -p "$dmg_contents"
  cp -R "$app_bundle" "$dmg_contents/"
  cp -f "$license_file" "$dmg_contents/" || true
  ln -s /Applications "$dmg_contents/Applications"
  hdiutil create -volname "XeniOS" -srcfolder "$dmg_contents" -ov -format UDZO "$dmg_out"
  rm -rf "$dmg_contents"
}

package_ios_ipa() {
  local app_bundle="$1"
  local ipa_out="$2"

  # Make the output absolute because we `cd` into a temp directory.
  local ipa_dir ipa_base
  ipa_dir="$(cd "$(dirname "$ipa_out")" && pwd)"
  ipa_base="$(basename "$ipa_out")"
  ipa_out="$ipa_dir/$ipa_base"

  rm -f "$ipa_out"
  local tmp
  tmp="$(mktemp -d "${TMPDIR:-/tmp}/xenia_ipa.XXXXXX")"
  mkdir -p "$tmp/Payload"
  # ditto preserves bundle metadata better than cp -R on macOS.
  ditto "$app_bundle" "$tmp/Payload/$(basename "$app_bundle")"
  (cd "$tmp" && ditto -c -k --sequesterRsrc --keepParent "Payload" "$ipa_out")
  rm -rf "$tmp"
}

print_metadata_summary() {
  local channel="$1"
  local stage="$2"
  local version="$3"
  local build_number="$4"
  local commit_short="$5"
  local issued_at="$6"
  local key_id="$7"
  local attestation_key="$8"

  local ios_build_id macos_build_id
  ios_build_id="$(make_build_id "ios" "$channel" "$version" "$build_number" "$stage" || true)"
  macos_build_id="$(make_build_id "macos" "$channel" "$version" "$build_number" "$stage" || true)"
  local version_build_label="$version"
  if [ -n "$build_number" ]; then
    version_build_label="${version}-${build_number}"
  fi

  echo "channel=$channel"
  echo "stage=$stage"
  echo "version=$version"
  echo "build_number=$build_number"
  echo "commit_short=$commit_short"
  echo "issued_at=$issued_at"
  if [ "$stage" = "stable" ]; then
    if [ "$channel" = "preview" ]; then
      echo "display_label=Preview $version_build_label"
    else
      echo "display_label=$version_build_label"
    fi
  else
    local stage_title
    case "$stage" in
      rc)
        stage_title="RC"
        ;;
      *)
        stage_title="$(printf '%s' "$stage" | awk '{print toupper(substr($0,1,1)) substr($0,2)}')"
        ;;
    esac
    if [ "$channel" = "preview" ]; then
      echo "display_label=$stage_title Preview $version_build_label"
    else
      echo "display_label=$stage_title $version_build_label"
    fi
  fi
  echo "ios_build_id=$ios_build_id"
  echo "macos_build_id=$macos_build_id"
  if [ -n "$attestation_key" ]; then
    local ios_payload ios_signature
    ios_payload="$(build_attestation_payload "ios" "$channel" "$ios_build_id" "$version" \
      "$build_number" "$stage" "$commit_short" "$issued_at" "$key_id")"
    ios_signature="$(sign_attestation_payload "$ios_payload" "$attestation_key")"
    echo "attestation=enabled"
    echo "attestation_key_id=$key_id"
    echo "ios_attestation_payload=$ios_payload"
    echo "ios_attestation_signature=$ios_signature"
  else
    echo "attestation=disabled"
  fi
}

build_ios=1
build_macos_arm64=1
build_macos_x86_64=1
print_metadata_only=0
out_dir="scratch/artifacts"
config="release"
release_version="${XENIOS_BUILD_VERSION:-}"
release_build_number="${XENIOS_BUILD_NUMBER:-}"
build_channel="${XENIOS_BUILD_CHANNEL:-release}"
release_stage="${XENIOS_BUILD_STAGE:-}"
stamp_bundle=""
stamp_platform=""
ios_min="16.0"
macos_min="15.0"
mac_sign_identity="-"
attestation_key="${XENIOS_BUILD_ATTESTATION_KEY:-}"
attestation_key_id="${XENIOS_BUILD_ATTESTATION_KEY_ID:-ci-hmac-v1}"

while [ $# -gt 0 ]; do
  case "$1" in
    --out)
      out_dir="${2:-}"; shift 2;;
    --config)
      config="${2:-}"; shift 2;;
    --version)
      release_version="${2:-}"; shift 2;;
    --build-number)
      release_build_number="${2:-}"; shift 2;;
    --channel)
      build_channel="${2:-}"; shift 2;;
    --stage)
      release_stage="${2:-}"; shift 2;;
    --attestation-key)
      attestation_key="${2:-}"; shift 2;;
    --attestation-key-file)
      [ -n "${2:-}" ] || die "missing value for --attestation-key-file"
      attestation_key="$(tr -d '\r\n' < "${2}")"; shift 2;;
    --stamp-bundle)
      stamp_bundle="${2:-}"; shift 2;;
    --platform)
      stamp_platform="${2:-}"; shift 2;;
    --attestation-key-id)
      attestation_key_id="${2:-}"; shift 2;;
    --print-metadata)
      print_metadata_only=1; shift;;
    --ios-min)
      ios_min="${2:-}"; shift 2;;
    --macos-min)
      macos_min="${2:-}"; shift 2;;
    --mac-sign)
      mac_sign_identity="${2:-}"; shift 2;;
    --skip-ios)
      build_ios=0; shift;;
    --skip-macos-arm64)
      build_macos_arm64=0; shift;;
    --skip-macos-x86_64)
      build_macos_x86_64=0; shift;;
    -h|--help)
      usage; exit 0;;
    *)
      die "unknown argument: $1";;
  esac
done

root="$(repo_root)"
cd "$root"

need_bin openssl

case "$build_channel" in
  release|preview)
    ;;
  *)
    die "build channel must be release or preview"
    ;;
esac

release_version="$(resolve_release_version "$root" "$release_version")"
release_stage="$(resolve_release_stage "$release_stage")"
release_build_number="$(resolve_release_build_number "$release_build_number")"

if [ -n "$attestation_key" ]; then
  [ -n "$release_version" ] || die "official attestation requires a marketing version (tag the repo or pass --version)"
  [ -n "$release_build_number" ] || die "official attestation requires a build number"
fi

mkdir -p "$out_dir"

buildcfg="$(cap_config "$config")"
commit_short="$(git rev-parse --short=9 HEAD)"
issued_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

if [ "$print_metadata_only" -eq 1 ]; then
  print_metadata_summary "$build_channel" "$release_stage" "$release_version" \
    "$release_build_number" "$commit_short" "$issued_at" "$attestation_key_id" "$attestation_key"
  exit 0
fi

if [ -n "$stamp_bundle" ]; then
  case "$stamp_platform" in
    ios|macos)
      ;;
    *)
      die "--platform must be ios or macos when using --stamp-bundle"
      ;;
  esac

  need_file_exec /usr/libexec/PlistBuddy
  stamp_bundle_version_metadata "$stamp_bundle" "$release_version" "$release_build_number"
  stamp_bundle_stage_metadata "$stamp_bundle" "$release_stage"
  stamp_bundle_attestation "$stamp_bundle" "$stamp_platform" "$build_channel" "$release_version" \
    "$release_build_number" "$release_stage" "$commit_short" "$issued_at" "$attestation_key_id" "$attestation_key"
  exit 0
fi

# Avoid stale iOS sentinel state leaking into normal macOS premake runs.
rm -f .ios_target

if [ "$(uname -s)" != "Darwin" ]; then
  die "this script must be run on macOS"
fi

need_bin xcodebuild
need_bin xcrun
need_bin codesign
need_bin hdiutil
need_bin ditto
need_bin xattr
need_bin otool
need_file_exec /usr/libexec/PlistBuddy

echo "Config: $buildcfg"
echo "Output: $out_dir"
if [ -n "$release_version" ] && [ -n "$release_build_number" ]; then
  echo "Bundle version: $release_version ($release_build_number)"
fi
echo "Build channel: $build_channel"
echo "Release stage: $release_stage"
echo "macOS min: $macos_min"
echo "iOS min: $ios_min"
echo "macOS signing: $mac_sign_identity"
if [ -n "$attestation_key" ]; then
  echo "Attestation: enabled (${attestation_key_id})"
else
  echo "Attestation: disabled"
fi

if [ "$build_macos_arm64" -eq 1 ]; then
  echo ""
  echo "== macOS arm64 =="
  ./xb build --config="$buildcfg" --arch=arm64 --target=xenia-app -- \
    CODE_SIGNING_ALLOWED=NO \
    MACOSX_DEPLOYMENT_TARGET="$macos_min"

  mac_dir="build/bin/Mac-ARM64/$buildcfg"
  app_bundle="$(find_first_app "$mac_dir")" || die "macOS arm64 app not found in $mac_dir"

  deploy_qt_if_missing "$app_bundle"

  if [ -f third_party/metal-shader-converter/lib/libmetalirconverter.dylib ]; then
    mkdir -p "$app_bundle/Contents/Frameworks"
    cp -f third_party/metal-shader-converter/lib/libmetalirconverter.dylib \
      "$app_bundle/Contents/Frameworks/" || true
  fi
  if [ -f third_party/DirectXShaderCompiler/build_dxilconv_macos/lib/libdxilconv.dylib ]; then
    mkdir -p "$app_bundle/Contents/Frameworks"
    cp -f third_party/DirectXShaderCompiler/build_dxilconv_macos/lib/libdxilconv.dylib \
      "$app_bundle/Contents/Frameworks/" || true
  fi

  stamp_bundle_version_metadata "$app_bundle" "$release_version" "$release_build_number"
  stamp_bundle_stage_metadata "$app_bundle" "$release_stage"
  compile_bundle_icon_assets "$app_bundle" "macosx" "$macos_min"
  stamp_bundle_attestation "$app_bundle" "macos" "$build_channel" "$release_version" \
    "$release_build_number" "$release_stage" "$commit_short" "$issued_at" "$attestation_key_id" "$attestation_key"
  require_sdl2_deployment_target "$app_bundle" "$macos_min"

  sign_macos_app "$app_bundle" "xenia.entitlements" "$mac_sign_identity"
  package_macos_dmg "$app_bundle" "$out_dir/xenios_macos_apple_silicon.dmg" "LICENSE"
fi

if [ "$build_macos_x86_64" -eq 1 ]; then
  echo ""
  echo "== macOS x86_64 =="
  ./xb build --config="$buildcfg" --arch=x86_64 --target=xenia-app -- \
    CODE_SIGNING_ALLOWED=NO \
    MACOSX_DEPLOYMENT_TARGET="$macos_min"

  mac_dir="build/bin/Mac-x86_64/$buildcfg"
  app_bundle="$(find_first_app "$mac_dir")" || die "macOS x86_64 app not found in $mac_dir"

  deploy_qt_if_missing "$app_bundle"

  if [ -f third_party/metal-shader-converter/lib/libmetalirconverter.dylib ]; then
    mkdir -p "$app_bundle/Contents/Frameworks"
    cp -f third_party/metal-shader-converter/lib/libmetalirconverter.dylib \
      "$app_bundle/Contents/Frameworks/" || true
  fi
  if [ -f third_party/DirectXShaderCompiler/build_dxilconv_macos_x86_64/lib/libdxilconv.dylib ]; then
    mkdir -p "$app_bundle/Contents/Frameworks"
    cp -f third_party/DirectXShaderCompiler/build_dxilconv_macos_x86_64/lib/libdxilconv.dylib \
      "$app_bundle/Contents/Frameworks/" || true
  fi

  stamp_bundle_version_metadata "$app_bundle" "$release_version" "$release_build_number"
  stamp_bundle_stage_metadata "$app_bundle" "$release_stage"
  compile_bundle_icon_assets "$app_bundle" "macosx" "$macos_min"
  stamp_bundle_attestation "$app_bundle" "macos" "$build_channel" "$release_version" \
    "$release_build_number" "$release_stage" "$commit_short" "$issued_at" "$attestation_key_id" "$attestation_key"
  require_sdl2_deployment_target "$app_bundle" "$macos_min"

  sign_macos_app "$app_bundle" "xenia.entitlements" "$mac_sign_identity"
  package_macos_dmg "$app_bundle" "$out_dir/xenios_macos_intel.dmg" "LICENSE"
fi

if [ "$build_ios" -eq 1 ]; then
  echo ""
  echo "== iOS arm64 (ad-hoc-signed ipa) =="

  ./xb premake --target_os=ios
  ./xb build --config="$buildcfg" --target_os=ios --no_premake --target=xenia-app -- \
    CODE_SIGNING_ALLOWED=NO \
    CODE_SIGNING_REQUIRED=NO \
    CODE_SIGN_STYLE=Manual \
    DEVELOPMENT_TEAM= \
    PROVISIONING_PROFILE_SPECIFIER= \
    CODE_SIGN_IDENTITY= \
    IPHONEOS_DEPLOYMENT_TARGET="$ios_min"

  ios_dir="build/bin/iOS-ARM64/$buildcfg"
  app_bundle="$(find_first_app "$ios_dir")" || die "iOS app not found in $ios_dir"

  stamp_bundle_version_metadata "$app_bundle" "$release_version" "$release_build_number"
  stamp_bundle_stage_metadata "$app_bundle" "$release_stage"
  compile_bundle_icon_assets "$app_bundle" "iphoneos" "$ios_min"
  stamp_bundle_attestation "$app_bundle" "ios" "$build_channel" "$release_version" \
    "$release_build_number" "$release_stage" "$commit_short" "$issued_at" "$attestation_key_id" "$attestation_key"
  # Ad-hoc sign to embed entitlements (increased-memory-limit).
  # Re-signing tools will preserve these when applying a real identity.
  codesign --force --sign - --entitlements "$root/xenia_ios.entitlements" "$app_bundle"

  package_ios_ipa "$app_bundle" "$out_dir/xenios_ios_iphone_ipad.ipa"
fi

echo ""
echo "Done. Artifacts:"
find "$out_dir" -maxdepth 1 -type f \( -name "*.dmg" -o -name "*.ipa" \) -print
