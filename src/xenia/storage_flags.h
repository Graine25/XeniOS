/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_STORAGE_FLAGS_H_
#define XENIA_STORAGE_FLAGS_H_

#include "xenia/base/cvar.h"

DECLARE_path(storage_root);
DECLARE_path(content_root);
DECLARE_path(cache_root);
DECLARE_bool(portable);
DECLARE_bool(mount_scratch);
DECLARE_bool(mount_cache);

#endif  // XENIA_STORAGE_FLAGS_H_
