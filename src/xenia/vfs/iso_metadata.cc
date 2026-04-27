/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/vfs/iso_metadata.h"

#include <cstring>
#include <vector>

#include "xenia/base/filesystem.h"
#include "xenia/base/mapped_memory.h"
#include "xenia/kernel/util/xex2_info.h"
#include "xenia/vfs/gdfx_util.h"

namespace xe {
namespace vfs {

namespace {

bool ReadIsoBytes(xe::filesystem::FileHandle* file, size_t image_size,
                  size_t offset, void* buffer, size_t length) {
  if (!file || !buffer) {
    return false;
  }
  if (length == 0) {
    return true;
  }
  if (offset > image_size || length > (image_size - offset)) {
    return false;
  }

  size_t bytes_read = 0;
  return file->Read(offset, buffer, length, &bytes_read) &&
         bytes_read == length;
}

std::optional<GdfxPartitionInfo> FindIsoPartitionFromFile(
    xe::filesystem::FileHandle* file, size_t image_size) {
  for (size_t offset : kGdfxLikelyOffsets) {
    const size_t sector32_offset = offset + (32 * kGdfxSectorSize);
    if (sector32_offset > image_size || image_size - sector32_offset < 28) {
      continue;
    }

    uint8_t sector32_data[28] = {};
    if (!ReadIsoBytes(file, image_size, sector32_offset, sector32_data,
                      sizeof(sector32_data))) {
      continue;
    }
    if (std::memcmp(sector32_data, kGdfxMagic, kGdfxMagicSize) != 0) {
      continue;
    }

    uint32_t root_sector = xe::load<uint32_t>(sector32_data + 20);
    uint32_t root_size = xe::load<uint32_t>(sector32_data + 24);
    if (root_size < 13 || root_size > 32 * 1024 * 1024) {
      continue;
    }

    return GdfxPartitionInfo{offset, root_sector, root_size};
  }

  return std::nullopt;
}

std::optional<GdfxFileLocation> FindIsoFileInDirectory(
    const uint8_t* directory_data, size_t directory_size,
    uint16_t entry_ordinal, const char* target_name, size_t target_name_length,
    int depth) {
  if (!directory_data || !target_name || depth > 100) {
    return std::nullopt;
  }

  size_t entry_offset = static_cast<size_t>(entry_ordinal) * 4;
  if (entry_offset + 14 > directory_size) {
    return std::nullopt;
  }

  const uint8_t* entry = directory_data + entry_offset;
  uint16_t node_left = xe::load<uint16_t>(entry + 0);
  uint16_t node_right = xe::load<uint16_t>(entry + 2);
  uint32_t sector = xe::load<uint32_t>(entry + 4);
  uint32_t length = xe::load<uint32_t>(entry + 8);
  uint8_t attributes = entry[12];
  uint8_t name_length = entry[13];

  if (entry_offset + 14 + name_length > directory_size) {
    return std::nullopt;
  }

  const char* name = reinterpret_cast<const char*>(entry + 14);
  if (node_left != 0) {
    auto left_result =
        FindIsoFileInDirectory(directory_data, directory_size, node_left,
                               target_name, target_name_length, depth + 1);
    if (left_result) {
      return left_result;
    }
  }

  const bool is_directory = (attributes & kGdfxFileAttributeDirectory) != 0;
  if (!is_directory && GdfxEqualsIgnoreCase(name, name_length, target_name,
                                            target_name_length)) {
    return GdfxFileLocation{
        static_cast<size_t>(sector) * kGdfxSectorSize,
        static_cast<size_t>(length),
    };
  }

  if (node_right != 0) {
    auto right_result =
        FindIsoFileInDirectory(directory_data, directory_size, node_right,
                               target_name, target_name_length, depth + 1);
    if (right_result) {
      return right_result;
    }
  }

  return std::nullopt;
}

std::optional<GdfxFileLocation> FindIsoFileFromFile(
    xe::filesystem::FileHandle* file, size_t image_size,
    const GdfxPartitionInfo& partition, const char* filename) {
  if (!filename) {
    return std::nullopt;
  }

  const size_t root_offset =
      partition.game_offset +
      (static_cast<size_t>(partition.root_sector) * kGdfxSectorSize);
  if (root_offset > image_size ||
      partition.root_size > (image_size - root_offset)) {
    return std::nullopt;
  }

  std::vector<uint8_t> root_directory(partition.root_size);
  if (!ReadIsoBytes(file, image_size, root_offset, root_directory.data(),
                    root_directory.size())) {
    return std::nullopt;
  }

  auto location =
      FindIsoFileInDirectory(root_directory.data(), root_directory.size(), 0,
                             filename, std::strlen(filename), 0);
  if (!location) {
    return std::nullopt;
  }

  location->offset += partition.game_offset;
  return location;
}

std::optional<XexMetadata> ExtractXexMetadataFromFileSlice(
    xe::filesystem::FileHandle* file, size_t image_size,
    const GdfxFileLocation& location) {
  if (location.length < sizeof(xex2_header)) {
    return std::nullopt;
  }

  xex2_header base_header = {};
  if (!ReadIsoBytes(file, image_size, location.offset, &base_header,
                    sizeof(base_header))) {
    return std::nullopt;
  }

  const uint32_t magic = base_header.magic;
  if (magic != 0x58455831 && magic != 0x58455832) {
    return std::nullopt;
  }

  const uint32_t header_size = base_header.header_size;
  if (header_size < sizeof(xex2_header) || header_size > location.length ||
      header_size > (image_size - location.offset)) {
    return std::nullopt;
  }

  std::vector<uint8_t> header_data(header_size);
  if (!ReadIsoBytes(file, image_size, location.offset, header_data.data(),
                    header_data.size())) {
    return std::nullopt;
  }

  return ExtractXexMetadata(header_data.data(), header_data.size());
}

}  // namespace

std::optional<XexMetadata> ExtractIsoMetadata(const uint8_t* data,
                                              size_t size) {
  // Find the game partition.
  auto partition = GdfxFindPartition(data, size);
  if (!partition) {
    return std::nullopt;
  }

  // Find default.xex in the root directory.
  auto location = GdfxFindFile(data, size, *partition, "default.xex");
  if (!location) {
    return std::nullopt;
  }

  // Validate the file location.
  if (location->offset + location->length > size) {
    return std::nullopt;
  }

  // Extract XEX metadata from the file data.
  return ExtractXexMetadata(data + location->offset, location->length);
}
std::optional<XexMetadata> ExtractIsoMetadata(
    const std::filesystem::path& path) {
  // Memory-map the ISO for efficient access.
  auto mmap = xe::MappedMemory::Open(path, xe::MappedMemory::Mode::kRead);
  if (!mmap) {
    auto file_info = xe::filesystem::GetInfo(path);
    auto file = xe::filesystem::FileHandle::OpenExisting(
        path, xe::filesystem::FileAccess::kGenericRead);
    if (!file_info ||
        file_info->type != xe::filesystem::FileInfo::Type::kFile || !file) {
      return std::nullopt;
    }

    auto partition =
        FindIsoPartitionFromFile(file.get(), file_info->total_size);
    if (!partition) {
      return std::nullopt;
    }

    auto location = FindIsoFileFromFile(file.get(), file_info->total_size,
                                        *partition, "default.xex");
    if (!location) {
      return std::nullopt;
    }

    return ExtractXexMetadataFromFileSlice(file.get(), file_info->total_size,
                                           *location);
  }

  return ExtractIsoMetadata(mmap->data(), mmap->size());
}

}  // namespace vfs
}  // namespace xe
