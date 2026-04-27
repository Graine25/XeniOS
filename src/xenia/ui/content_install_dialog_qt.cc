/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/ui/content_install_dialog_qt.h"

#include <QDesktopServices>
#include <QFrame>
#include <QGridLayout>
#include <QHeaderView>
#include <QImage>
#include <QMessageBox>
#include <QPixmap>
#include <QScrollArea>
#include <QUrl>

#include "third_party/fmt/include/fmt/format.h"
#include "xenia/base/system.h"
#include "xenia/ui/qt_util.h"
#include "xenia/xbox.h"

namespace xe {
namespace ui {

using xe::ui::SafeQString;

ContentInstallDialogQt::ContentInstallDialogQt(
    QWidget* parent, const std::filesystem::path& content_root,
    std::shared_ptr<std::vector<Emulator::ContentInstallEntry>> entries)
    : QDialog(parent),
      content_root_(content_root),
      installation_entries_(entries),
      is_zarchive_mode_(false) {
  SetupUI();

  update_timer_ = new QTimer(this);
  connect(update_timer_, &QTimer::timeout, this,
          &ContentInstallDialogQt::UpdateProgress);
  update_timer_->start(100);
}

ContentInstallDialogQt::ContentInstallDialogQt(
    QWidget* parent,
    std::shared_ptr<std::vector<Emulator::ZarchiveEntry>> entries)
    : QDialog(parent), zarchive_entries_(entries), is_zarchive_mode_(true) {
  SetupUI();

  update_timer_ = new QTimer(this);
  connect(update_timer_, &QTimer::timeout, this,
          &ContentInstallDialogQt::UpdateProgress);
  update_timer_->start(100);
}

ContentInstallDialogQt::~ContentInstallDialogQt() {
  if (update_timer_) {
    update_timer_->stop();
  }
}

size_t ContentInstallDialogQt::GetEntryCount() const {
  if (is_zarchive_mode_) {
    return zarchive_entries_ ? zarchive_entries_->size() : 0;
  }
  return installation_entries_ ? installation_entries_->size() : 0;
}

void ContentInstallDialogQt::SetupUI() {
  if (is_zarchive_mode_) {
    setWindowTitle("Zarchive Progress");
  } else {
    setWindowTitle("Installation Progress");
  }
  setModal(true);
  setAttribute(Qt::WA_DeleteOnClose);

  auto* main_layout = new QVBoxLayout(this);

  // Create table widget with 1 column per entry
  table_widget_ = new QTableWidget(this);
  table_widget_->setColumnCount(1);
  table_widget_->setRowCount(static_cast<int>(GetEntryCount()));
  table_widget_->horizontalHeader()->setVisible(false);
  table_widget_->verticalHeader()->setVisible(false);
  table_widget_->setShowGrid(false);
  table_widget_->setSelectionMode(QAbstractItemView::NoSelection);
  table_widget_->setEditTriggers(QAbstractItemView::NoEditTriggers);
  table_widget_->horizontalHeader()->setStretchLastSection(true);

  // Populate table for each entry
  for (size_t i = 0; i < GetEntryCount(); i++) {
    // Create a container widget for the entire row
    auto* row_widget = new QWidget();
    auto* row_layout = new QHBoxLayout(row_widget);
    row_layout->setContentsMargins(10, 10, 10, 10);
    row_layout->setSpacing(15);

    // Icon label - will be sized dynamically based on row height
    auto* icon_label = new QLabel();
    icon_label->setMinimumSize(80, 80);
    icon_label->setAlignment(Qt::AlignCenter);

    if (!is_zarchive_mode_) {
      auto& entry = (*installation_entries_)[i];
      // Load icon from PNG data if available
      if (!entry.icon_data_.empty()) {
        QImage image;
        if (image.loadFromData(entry.icon_data_.data(),
                               static_cast<int>(entry.icon_data_.size()))) {
          // Scale pixmap to a larger size while preserving aspect ratio
          QPixmap pixmap = QPixmap::fromImage(image);
          icon_label->setPixmap(pixmap.scaled(80, 80, Qt::KeepAspectRatio,
                                              Qt::SmoothTransformation));
        } else {
          // Fallback if PNG loading fails
          icon_label->setText("pkg");
          QFont icon_font = icon_label->font();
          icon_font.setPointSize(icon_font.pointSize() * 2);
          icon_label->setFont(icon_font);
        }
      } else {
        // No icon data, show placeholder
        icon_label->setText("pkg");
        QFont icon_font = icon_label->font();
        icon_font.setPointSize(icon_font.pointSize() * 2);
        icon_label->setFont(icon_font);
      }
    } else {
      // Zarchive mode - try to load icon from entry
      auto& entry = (*zarchive_entries_)[i];
      if (!entry.icon_data_.empty()) {
        QImage image;
        if (image.loadFromData(entry.icon_data_.data(),
                               static_cast<int>(entry.icon_data_.size()))) {
          QPixmap pixmap = QPixmap::fromImage(image);
          icon_label->setPixmap(pixmap.scaled(80, 80, Qt::KeepAspectRatio,
                                              Qt::SmoothTransformation));
        } else {
          // Fallback if loading fails
          icon_label->setText("zar");
          QFont icon_font = icon_label->font();
          icon_font.setPointSize(icon_font.pointSize() * 2);
          icon_label->setFont(icon_font);
        }
      } else {
        // No icon data, show placeholder
        icon_label->setText("zar");
        QFont icon_font = icon_label->font();
        icon_font.setPointSize(icon_font.pointSize() * 2);
        icon_label->setFont(icon_font);
      }
    }
    row_layout->addWidget(icon_label);

    // Info column
    auto* info_widget = new QWidget();
    auto* info_layout = new QVBoxLayout(info_widget);
    info_layout->setContentsMargins(0, 0, 0, 0);
    info_layout->setSpacing(2);

    auto* name_label = new QLabel();
    name_label->setTextFormat(Qt::PlainText);
    info_layout->addWidget(name_label);

    auto* path_label = new QLabel();
    path_label->setTextFormat(Qt::RichText);
    path_label->setTextInteractionFlags(Qt::TextBrowserInteraction);
    path_label->setOpenExternalLinks(false);

    if (!is_zarchive_mode_) {
      connect(path_label, &QLabel::linkActivated, this,
              [this, i](const QString&) {
                auto& entry = (*installation_entries_)[i];
                // Launch file explorer in a separate thread to avoid blocking
                // the UI
                std::thread path_open(
                    LaunchFileExplorer,
                    content_root_ / entry.data_installation_path_);
                path_open.detach();
              });
    } else {
      connect(
          path_label, &QLabel::linkActivated, this, [this, i](const QString&) {
            auto& entry = (*zarchive_entries_)[i];
            // Launch file explorer in a separate thread to avoid blocking the
            // UI
            std::thread path_open(LaunchFileExplorer,
                                  entry.data_installation_path_.parent_path());
            path_open.detach();
          });
    }
    info_layout->addWidget(path_label);

    auto* type_label = new QLabel();
    info_layout->addWidget(type_label);

    auto* status_label = new QLabel();
    info_layout->addWidget(status_label);

    auto* progress_bar = new QProgressBar();
    progress_bar->setMinimum(0);
    progress_bar->setMaximum(100);
    info_layout->addWidget(progress_bar);

    row_layout->addWidget(info_widget, 1);

    table_widget_->setCellWidget(static_cast<int>(i), 0, row_widget);

    // Auto-resize row height to fit content
    table_widget_->resizeRowToContents(static_cast<int>(i));

    // Store widgets for later updates
    entry_widgets_.push_back({icon_label, name_label, path_label, type_label,
                              status_label, progress_bar});
  }

  main_layout->addWidget(table_widget_);

  // Button layout
  auto* button_layout = new QHBoxLayout();

  // Cancel button
  cancel_button_ = new QPushButton("Cancel", this);
  connect(cancel_button_, &QPushButton::clicked, this,
          &ContentInstallDialogQt::OnCancelClicked);
  button_layout->addWidget(cancel_button_);

  button_layout->addStretch();

  // Close button
  close_button_ = new QPushButton("Close", this);
  close_button_->setEnabled(false);
  connect(close_button_, &QPushButton::clicked, this,
          &ContentInstallDialogQt::OnCloseClicked);
  button_layout->addWidget(close_button_);

  main_layout->addLayout(button_layout);

  setMinimumSize(600, 400);

  // Initial update
  UpdateProgress();
}

void ContentInstallDialogQt::UpdateProgress() {
  bool is_everything_installed = true;
  bool has_cancelled_entries = false;
  bool cancellation_complete = false;

  for (size_t i = 0; i < GetEntryCount(); i++) {
    auto& widgets = entry_widgets_[i];

    // Get entry fields based on mode
    std::string name;
    std::filesystem::path dest_path;
    Emulator::InstallState state;
    X_STATUS result;
    std::string error_message;
    uint64_t content_size;
    uint64_t current_size;
    bool cancelled;

    if (is_zarchive_mode_) {
      auto& entry = (*zarchive_entries_)[i];
      name = entry.name_;
      dest_path = entry.data_installation_path_;
      state = entry.installation_state_;
      result = entry.installation_result_;
      error_message = entry.installation_error_message_;
      content_size = entry.content_size_;
      current_size = entry.currently_installed_size_;
      cancelled = entry.cancelled_.load();
    } else {
      auto& entry = (*installation_entries_)[i];
      name = entry.name_;
      dest_path = entry.data_installation_path_;
      state = entry.installation_state_;
      result = entry.installation_result_;
      error_message = entry.installation_error_message_;
      content_size = entry.content_size_;
      current_size = entry.currently_installed_size_;
      cancelled = entry.cancelled_.load();
    }

    // Check if this entry was cancelled and has finished
    if (cancelled && state == xe::Emulator::InstallState::failed &&
        result == X_ERROR_CANCELLED) {
      has_cancelled_entries = true;
    }

    // Update name
    widgets.name_label->setText(QString("Name: %1").arg(SafeQString(name)));

    // Update path with link
    QString path_str = SafeQString(xe::path_to_utf8(dest_path));
    if (is_zarchive_mode_) {
      auto& entry = (*zarchive_entries_)[i];
      if (entry.operation_ == Emulator::ZarchiveOperation::Create) {
        widgets.path_label->setText(
            QString("Output: <a href=\"%1\">%1</a>").arg(path_str));
      } else {
        widgets.path_label->setText(
            QString("Extract To: <a href=\"%1\">%1</a>").arg(path_str));
      }
    } else {
      widgets.path_label->setText(
          QString("Installation Path: <a href=\"%1\">%1</a>").arg(path_str));
    }

    // Update type/operation label
    if (is_zarchive_mode_) {
      auto& entry = (*zarchive_entries_)[i];
      if (entry.operation_ == Emulator::ZarchiveOperation::Create) {
        widgets.type_label->setText("Operation: Create Archive");
      } else {
        widgets.type_label->setText("Operation: Extract Archive");
      }
    } else {
      auto& entry = (*installation_entries_)[i];
      if (entry.content_type_ != xe::XContentType::kInvalid) {
        auto it = XContentTypeMap.find(entry.content_type_);
        if (it != XContentTypeMap.end()) {
          widgets.type_label->setText(
              QString("Content Type: %1").arg(SafeQString(it->second)));
        }
      } else {
        widgets.type_label->setText("");
      }
    }

    // Update status
    std::string status_str = fmt::format(
        "Status: {}",
        xe::Emulator::installStateStringName[static_cast<uint8_t>(state)]);

    if (state == xe::Emulator::InstallState::failed) {
      status_str +=
          fmt::format(" - {} ({:08X})", error_message.c_str(), result);
    }

    widgets.status_label->setText(SafeQString(status_str));

    // Update progress bar
    if (content_size > 0) {
      int progress = static_cast<int>(
          (static_cast<double>(current_size) / content_size) * 100.0);
      widgets.progress_bar->setValue(progress);

      if (current_size != content_size && result == X_ERROR_SUCCESS &&
          !cancelled) {
        is_everything_installed = false;
      }
    } else {
      widgets.progress_bar->setValue(0);
      // If content_size is still 0, hasn't been processed yet
      if (state != Emulator::InstallState::failed && !cancelled) {
        is_everything_installed = false;
      }
    }
  }

  // Check if cancellation is complete
  if (has_cancelled_entries && !cancel_button_->isEnabled()) {
    bool all_cancelled_complete = true;
    if (is_zarchive_mode_) {
      for (const auto& entry : *zarchive_entries_) {
        if (entry.cancelled_.load()) {
          if (entry.installation_state_ != xe::Emulator::InstallState::failed) {
            all_cancelled_complete = false;
            break;
          }
        }
      }
    } else {
      for (const auto& entry : *installation_entries_) {
        if (entry.cancelled_.load()) {
          if (entry.installation_state_ != xe::Emulator::InstallState::failed) {
            all_cancelled_complete = false;
            break;
          }
        }
      }
    }
    cancellation_complete = all_cancelled_complete;
  }

  // Show cancellation dialog and remove cancelled entries
  if (cancellation_complete && !cancellation_dialog_shown_) {
    cancellation_dialog_shown_ = true;

    // Count cancelled entries
    int cancelled_count = 0;
    if (is_zarchive_mode_) {
      for (const auto& entry : *zarchive_entries_) {
        if (entry.cancelled_.load() &&
            entry.installation_result_ == X_ERROR_CANCELLED) {
          cancelled_count++;
        }
      }
    } else {
      for (const auto& entry : *installation_entries_) {
        if (entry.cancelled_.load() &&
            entry.installation_result_ == X_ERROR_CANCELLED) {
          cancelled_count++;
        }
      }
    }

    // Show dialog
    QString message =
        is_zarchive_mode_
            ? QString("Cancelled %1 operation(s).").arg(cancelled_count)
            : QString(
                  "Cancelled %1 installation(s) and cleaned up partial files.")
                  .arg(cancelled_count);
    QMessageBox::information(this, "Cancelled", message);

    // Remove cancelled entries from the vector and UI
    if (is_zarchive_mode_) {
      for (int i = static_cast<int>(zarchive_entries_->size()) - 1; i >= 0;
           i--) {
        auto& entry = (*zarchive_entries_)[i];
        if (entry.cancelled_.load() &&
            entry.installation_result_ == X_ERROR_CANCELLED) {
          table_widget_->removeRow(i);
          entry_widgets_.erase(entry_widgets_.begin() + i);
          zarchive_entries_->erase(zarchive_entries_->begin() + i);
        }
      }
    } else {
      for (int i = static_cast<int>(installation_entries_->size()) - 1; i >= 0;
           i--) {
        auto& entry = (*installation_entries_)[i];
        if (entry.cancelled_.load() &&
            entry.installation_result_ == X_ERROR_CANCELLED) {
          table_widget_->removeRow(i);
          entry_widgets_.erase(entry_widgets_.begin() + i);
          installation_entries_->erase(installation_entries_->begin() + i);
        }
      }
    }

    // Reset cancel button
    cancel_button_->setText("Cancel");
    cancel_button_->setEnabled(true);
    cancellation_dialog_shown_ = false;
  }

  // Enable close button when everything is done, disable cancel
  close_button_->setEnabled(is_everything_installed);
  if (!cancellation_complete) {
    cancel_button_->setEnabled(!is_everything_installed);
  }

  // Show success dialog when all operations complete (detect transition)
  bool has_entries = is_zarchive_mode_ ? !zarchive_entries_->empty()
                                       : !installation_entries_->empty();
  if (is_everything_installed && !was_everything_installed_ &&
      !success_dialog_shown_ && has_entries) {
    success_dialog_shown_ = true;

    // Count successful operations
    int success_count = 0;
    if (is_zarchive_mode_) {
      for (const auto& entry : *zarchive_entries_) {
        if (entry.installation_state_ == Emulator::InstallState::installed) {
          success_count++;
        }
      }
    } else {
      for (const auto& entry : *installation_entries_) {
        if (entry.installation_state_ == Emulator::InstallState::installed) {
          success_count++;
        }
      }
    }

    // Show dialog immediately on completion
    if (success_count > 0) {
      QString message = is_zarchive_mode_
                            ? QString("Successfully completed %1 operation(s)!")
                                  .arg(success_count)
                            : QString("Successfully installed %1 package(s)!")
                                  .arg(success_count);
      QMessageBox::information(this, "Complete", message);
    }
  }

  // Track previous state for next update
  was_everything_installed_ = is_everything_installed;

  // Auto-close if everything is done and user closed the dialog
  if (is_everything_installed && !isVisible()) {
    deleteLater();
  }
}

bool ContentInstallDialogQt::IsEverythingInstalled() {
  if (is_zarchive_mode_) {
    for (const auto& entry : *zarchive_entries_) {
      if (entry.content_size_ > 0) {
        if (entry.currently_installed_size_ != entry.content_size_ &&
            entry.installation_result_ == X_ERROR_SUCCESS &&
            !entry.cancelled_.load()) {
          return false;
        }
      } else {
        if (entry.installation_state_ != Emulator::InstallState::failed &&
            !entry.cancelled_.load()) {
          return false;
        }
      }
    }
  } else {
    for (const auto& entry : *installation_entries_) {
      if (entry.content_size_ > 0) {
        if (entry.currently_installed_size_ != entry.content_size_ &&
            entry.installation_result_ == X_ERROR_SUCCESS &&
            !entry.cancelled_.load()) {
          return false;
        }
      } else {
        if (entry.installation_state_ != Emulator::InstallState::failed &&
            !entry.cancelled_.load()) {
          return false;
        }
      }
    }
  }
  return true;
}

void ContentInstallDialogQt::OnCloseClicked() {
  if (IsEverythingInstalled()) {
    accept();
  }
}

void ContentInstallDialogQt::OnCancelClicked() {
  // Set the cancelled flag for all entries that are still in progress
  if (is_zarchive_mode_) {
    for (auto& entry : *zarchive_entries_) {
      if (entry.installation_state_ == Emulator::InstallState::preparing ||
          entry.installation_state_ == Emulator::InstallState::pending ||
          entry.installation_state_ == Emulator::InstallState::installing) {
        entry.cancelled_.store(true);
      }
    }
  } else {
    for (auto& entry : *installation_entries_) {
      if (entry.installation_state_ == Emulator::InstallState::preparing ||
          entry.installation_state_ == Emulator::InstallState::pending ||
          entry.installation_state_ == Emulator::InstallState::installing) {
        entry.cancelled_.store(true);
      }
    }
  }

  // Disable the cancel button
  cancel_button_->setEnabled(false);
  cancel_button_->setText("Cancelling...");
}

}  // namespace ui
}  // namespace xe
