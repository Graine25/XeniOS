/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_UI_CONTENT_INSTALL_DIALOG_QT_H_
#define XENIA_UI_CONTENT_INSTALL_DIALOG_QT_H_

#include <QDialog>
#include <QLabel>
#include <QProgressBar>
#include <QPushButton>
#include <QTableWidget>
#include <QTimer>
#include <QVBoxLayout>
#include <memory>
#include <variant>
#include <vector>

#include "xenia/emulator.h"

namespace xe {
namespace ui {

class ContentInstallDialogQt : public QDialog {
  Q_OBJECT

 public:
  ContentInstallDialogQt(
      QWidget* parent, const std::filesystem::path& content_root,
      std::shared_ptr<std::vector<Emulator::ContentInstallEntry>> entries);

  ContentInstallDialogQt(
      QWidget* parent,
      std::shared_ptr<std::vector<Emulator::ZarchiveEntry>> entries);

  ~ContentInstallDialogQt() override;

 private slots:
  void UpdateProgress();
  void OnCloseClicked();
  void OnCancelClicked();

 private:
  void SetupUI();
  bool IsEverythingInstalled();
  size_t GetEntryCount() const;

  std::filesystem::path content_root_;
  std::shared_ptr<std::vector<Emulator::ContentInstallEntry>>
      installation_entries_;
  std::shared_ptr<std::vector<Emulator::ZarchiveEntry>> zarchive_entries_;
  bool is_zarchive_mode_ = false;

  QTableWidget* table_widget_;
  QPushButton* close_button_;
  QPushButton* cancel_button_;
  QTimer* update_timer_;
  bool cancellation_dialog_shown_ = false;
  bool success_dialog_shown_ = false;
  bool was_everything_installed_ = false;

  struct EntryWidgets {
    QLabel* icon_label;
    QLabel* name_label;
    QLabel* path_label;
    QLabel* type_label;
    QLabel* status_label;
    QProgressBar* progress_bar;
  };

  std::vector<EntryWidgets> entry_widgets_;
};

}  // namespace ui
}  // namespace xe

#endif  // XENIA_UI_CONTENT_INSTALL_DIALOG_QT_H_
