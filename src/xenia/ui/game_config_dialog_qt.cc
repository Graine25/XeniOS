/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/ui/game_config_dialog_qt.h"

#include <QCheckBox>
#include <QComboBox>
#include <QFileDialog>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QMessageBox>
#include <QSpinBox>
#include <QVBoxLayout>
#include <fstream>
#include <limits>
#include <sstream>

#include "third_party/fmt/include/fmt/format.h"
#include "third_party/rapidjson/include/rapidjson/document.h"
#include "third_party/rapidjson/include/rapidjson/error/en.h"
#include "third_party/rapidjson/include/rapidjson/stringbuffer.h"
#include "third_party/rapidjson/include/rapidjson/writer.h"
#include "third_party/tomlplusplus/toml.hpp"
#include "xenia/app/emulator_window.h"
#include "xenia/base/cvar.h"
#include "xenia/base/filesystem.h"
#include "xenia/base/logging.h"
#include "xenia/config.h"
#include "xenia/ui/config_helpers.h"
#include "xenia/ui/qt_util.h"

namespace xe {
namespace app {

namespace {

// Use the shared enum options from config_helpers.h
using xe::ui::GetKnownEnumOptions;
using xe::ui::SafeQString;
using xe::ui::SafeStdString;

// Helper to convert a RapidJSON value to string
std::string JsonValueToString(const rapidjson::Value& value) {
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  value.Accept(writer);
  std::string result = buffer.GetString();

  // Remove quotes from strings since we're storing raw config values
  if (result.length() >= 2 && result.front() == '"' && result.back() == '"') {
    result = result.substr(1, result.length() - 2);
  }

  return result;
}

bool CanaryCvarCompat(const std::string& var_name, const std::string& value,
                      std::string& out_var_name, std::string& out_value) {
  for (const auto& alias : xe::ui::GetCvarAliases()) {
    if (var_name == alias.old_name && value == alias.old_value) {
      out_var_name = alias.new_name;
      out_value = alias.new_value;
      return true;
    }
  }

  out_var_name = var_name;
  out_value = value;
  return false;
}

}  // namespace

GameConfigDialogQt::GameConfigDialogQt(QWidget* parent,
                                       EmulatorWindow* emulator_window,
                                       uint32_t title_id,
                                       const std::string& game_title)
    : QDialog(parent),
      emulator_window_(emulator_window),
      title_id_(title_id),
      game_title_(game_title) {
  SetupUI();
  LoadConfigOverrides();

  // Check if recommended settings exist and enable/disable button accordingly
  auto settings_path = config::GetBundledDataPath("optimized_settings") /
                       (fmt::format("{:08X}", title_id_) + ".json");
  recommended_button_->setEnabled(std::filesystem::exists(settings_path));
}

GameConfigDialogQt::~GameConfigDialogQt() = default;

void GameConfigDialogQt::SetupUI() {
  setWindowTitle(
      SafeQString(fmt::format("Game Config Overrides - {}", game_title_)));
  setMinimumSize(800, 500);
  resize(900, 600);

  auto* main_layout = new QVBoxLayout(this);

  // Info label
  auto* info_label = new QLabel(
      "Add or modify configuration overrides for this game. These settings "
      "will only apply when launching this specific title.",
      this);
  info_label->setWordWrap(true);
  info_label->setStyleSheet("QLabel { color: gray; margin-bottom: 10px; }");
  main_layout->addWidget(info_label);

  // Table for config overrides
  overrides_table_ = new QTableWidget(this);
  overrides_table_->setColumnCount(3);
  overrides_table_->setHorizontalHeaderLabels({"Config Variable", "Value", ""});
  overrides_table_->horizontalHeader()->setStretchLastSection(false);
  overrides_table_->horizontalHeader()->setSectionResizeMode(
      0, QHeaderView::Interactive);
  overrides_table_->horizontalHeader()->setSectionResizeMode(
      1, QHeaderView::Stretch);
  overrides_table_->horizontalHeader()->setSectionResizeMode(
      2, QHeaderView::Fixed);
  overrides_table_->setColumnWidth(0, 300);
  overrides_table_->setColumnWidth(2, 40);  // Width for delete button
  overrides_table_->setSelectionBehavior(QAbstractItemView::SelectRows);
  overrides_table_->setSelectionMode(QAbstractItemView::SingleSelection);
  overrides_table_->setAlternatingRowColors(true);
  overrides_table_->setFocusPolicy(Qt::NoFocus);

  // Hide row numbers on the left
  overrides_table_->verticalHeader()->setVisible(false);

  // Remove selection highlighting and focus rectangles
  overrides_table_->setStyleSheet(
      "QTableWidget { selection-background-color: transparent; }"
      "QTableWidget::item:selected { background-color: transparent; }"
      "QTableWidget::item:focus { outline: none; border: none; }");

  // Disable editing on the table itself (widgets handle their own editing)
  overrides_table_->setEditTriggers(QAbstractItemView::NoEditTriggers);

  main_layout->addWidget(overrides_table_);

  // Buttons for add and recommended settings
  auto* button_layout = new QHBoxLayout();

  add_button_ = new QPushButton("Add Override", this);
  connect(add_button_, &QPushButton::clicked, this,
          &GameConfigDialogQt::OnAddOverrideClicked);

  recommended_button_ = new QPushButton("Load Recommended", this);
  recommended_button_->setToolTip(
      "Load optimized settings for this game from the community database");
  connect(recommended_button_, &QPushButton::clicked, this,
          &GameConfigDialogQt::OnUseRecommendedClicked);

  button_layout->addWidget(add_button_);
  button_layout->addWidget(recommended_button_);
  button_layout->addStretch();

  main_layout->addLayout(button_layout);

  // Dialog buttons (Save/Cancel)
  auto* dialog_button_layout = new QHBoxLayout();
  dialog_button_layout->addStretch();

  cancel_button_ = new QPushButton("Cancel", this);
  save_button_ = new QPushButton("Save", this);
  save_button_->setDefault(true);

  connect(cancel_button_, &QPushButton::clicked, this,
          &GameConfigDialogQt::OnCancelClicked);
  connect(save_button_, &QPushButton::clicked, this,
          &GameConfigDialogQt::OnSaveClicked);

  dialog_button_layout->addWidget(cancel_button_);
  dialog_button_layout->addWidget(save_button_);

  main_layout->addLayout(dialog_button_layout);
}

std::filesystem::path GameConfigDialogQt::GetGameConfigPath() const {
  return config::config_folder / "config" /
         (fmt::format("{:08X}", title_id_) + config::game_config_suffix);
}

void GameConfigDialogQt::LoadConfigOverrides() {
  config_overrides_.clear();
  original_overrides_.clear();
  overrides_table_->setRowCount(0);

  try {
    const auto config = config::LoadGameConfig(title_id_);

    if (!cvar::ConfigVars) {
      return;
    }

    // Iterate through all registered cvars to find ones in this config file
    for (auto& it : *cvar::ConfigVars) {
      auto config_var = static_cast<cvar::IConfigVar*>(it.second);
      toml::path config_key =
          toml::path(config_var->category() + "." + config_var->name());

      const auto config_key_node = config.at_path(config_key);
      if (config_key_node) {
        // Load the value into the cvar temporarily to validate and convert it
        config_var->LoadConfigValue(config_key_node.node());

        // Get the string representation from the cvar
        std::string value = config_var->config_value();

        // Strip TOML quotes from string values (they're added by ToString for
        // TOML format)
        if (value.length() >= 2 && value.front() == '"' &&
            value.back() == '"') {
          value = value.substr(1, value.length() - 2);
        }

        std::string var_name = config_var->name();
        config_overrides_[var_name] = value;
        original_overrides_[var_name] = value;  // Store original value

        // Add to table
        CreateRow(var_name, value);
      }
    }

    // Check for legacy cvar names in the TOML that need migration
    for (const auto& [category_name, category_table] : config) {
      if (!category_table.is_table()) continue;

      for (const auto& [key, value] : *category_table.as_table()) {
        std::string var_name = std::string(key);
        std::string var_value;

        // Handle different TOML value types
        if (value.is_boolean()) {
          var_value = value.as_boolean()->get() ? "true" : "false";
        } else {
          var_value = value.value_or("");
          if (var_value.length() >= 2 && var_value.front() == '"' &&
              var_value.back() == '"') {
            var_value = var_value.substr(1, var_value.length() - 2);
          }
        }

        std::string translated_name;
        std::string translated_value;
        if (CanaryCvarCompat(var_name, var_value, translated_name,
                             translated_value)) {
          if (translated_name != var_name &&
              !config_overrides_.count(translated_name)) {
            config_overrides_[translated_name] = translated_value;
            CreateRow(translated_name, translated_value);
            has_unsaved_changes_ = true;
          }
        }
      }
    }

    // Sort the table by cvar name after all entries are added
    overrides_table_->sortItems(0, Qt::AscendingOrder);
  } catch (const std::exception& e) {
    XELOGE("Failed to load game config for {:08X}: {}", title_id_, e.what());
    QMessageBox::warning(
        this, "Error Loading Config",
        SafeQString(fmt::format("Failed to load game config: {}", e.what())));
  }

  has_unsaved_changes_ = false;
}

void GameConfigDialogQt::SaveConfigOverrides() {
  // Build a TOML structure from current table contents
  toml::table config_table;
  std::map<std::string, toml::table> category_tables;

  for (int row = 0; row < overrides_table_->rowCount(); ++row) {
    auto* var_item = overrides_table_->item(row, 0);
    if (!var_item) {
      continue;
    }

    std::string var_name = SafeStdString(var_item->text());

    // Get value from the cell widget, not from a text item
    QWidget* value_widget = overrides_table_->cellWidget(row, 1);
    if (!value_widget) {
      continue;
    }

    std::string value_str = GetEditorValue(value_widget);

    if (var_name.empty()) {
      continue;
    }

    // Find the cvar to get its category and type
    auto* config_var =
        cvar::ConfigVars ? (*cvar::ConfigVars)[var_name] : nullptr;

    if (!config_var) {
      XELOGW("Unknown config variable: {}", var_name);
      continue;
    }

    std::string category = config_var->category();

    // Parse the value based on the config var type
    try {
      if (config_var->is_transient()) {
        // Skip transient cvars
        continue;
      }

      // Try to parse the value appropriately
      // First check if it's a boolean
      if (value_str == "true" || value_str == "false") {
        category_tables[category].insert_or_assign(var_name,
                                                   value_str == "true");
      } else {
        // Try to parse as integer
        char* end;
        long long int_val = std::strtoll(value_str.c_str(), &end, 10);
        if (*end == '\0') {
          category_tables[category].insert_or_assign(var_name, int_val);
        } else {
          // Try to parse as double
          double double_val = std::strtod(value_str.c_str(), &end);
          if (*end == '\0') {
            category_tables[category].insert_or_assign(var_name, double_val);
          } else {
            // Treat as string
            category_tables[category].insert_or_assign(var_name, value_str);
          }
        }
      }
    } catch (const std::exception& e) {
      XELOGE("Failed to parse value for {}: {}", var_name, e.what());
    }
  }

  // Build the final config table
  for (const auto& [category, table] : category_tables) {
    config_table.insert_or_assign(category, table);
  }

  // Save to file using config::SaveGameConfig
  try {
    config::SaveGameConfig(title_id_, config_table);

    has_unsaved_changes_ = false;

    QMessageBox::information(
        this, "Config Saved",
        "Game configuration overrides have been saved successfully.\n\n"
        "These settings will be applied the next time you launch this game.");
  } catch (const std::exception& e) {
    QMessageBox::critical(
        this, "Error Saving Config",
        SafeQString(fmt::format("Failed to save game config: {}", e.what())));
  }
}

QWidget* GameConfigDialogQt::CreateEditorForCvar(
    cvar::IConfigVar* var, const std::string& current_value) {
  std::string trimmed_current = current_value;
  trimmed_current.erase(0, trimmed_current.find_first_not_of(" \t\n\r"));
  trimmed_current.erase(trimmed_current.find_last_not_of(" \t\n\r") + 1);

  // Check if this is an enum-like cvar with known options
  const auto& enum_options = GetKnownEnumOptions();
  auto enum_it = enum_options.find(var->name());

  if (trimmed_current == "true" || trimmed_current == "false") {
    // Boolean dropdown (more explicit than checkbox for config overrides)
    auto* combo = new QComboBox();
    combo->addItem("true");
    combo->addItem("false");
    combo->setCurrentText(SafeQString(current_value));
    connect(combo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            [this, combo]() {
              has_unsaved_changes_ = true;
              // Find which row this widget belongs to and update its bold state
              for (int row = 0; row < overrides_table_->rowCount(); ++row) {
                if (overrides_table_->cellWidget(row, 1) == combo) {
                  UpdateRowModifiedState(row);
                  break;
                }
              }
            });
    return combo;
  } else if (enum_it != enum_options.end()) {
    // Dropdown for enum-like cvars
    auto* combo = new QComboBox();
    const auto& options = enum_it->second;

    int current_index = -1;
    for (size_t i = 0; i < options.size(); ++i) {
      combo->addItem(SafeQString(options[i]));
      if (options[i] == current_value) {
        current_index = static_cast<int>(i);
      }
    }

    if (current_index >= 0) {
      combo->setCurrentIndex(current_index);
    }

    connect(combo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            [this, combo]() {
              has_unsaved_changes_ = true;
              // Find which row this widget belongs to and update its bold state
              for (int row = 0; row < overrides_table_->rowCount(); ++row) {
                if (overrides_table_->cellWidget(row, 1) == combo) {
                  UpdateRowModifiedState(row);
                  break;
                }
              }
            });
    return combo;
  } else if (dynamic_cast<cvar::ConfigVar<std::filesystem::path>*>(var)) {
    // Path input with browse button
    auto* container = new QWidget();
    auto* layout = new QHBoxLayout(container);
    layout->setContentsMargins(0, 0, 0, 0);

    auto* line_edit = new QLineEdit();
    line_edit->setText(SafeQString(current_value));
    connect(line_edit, &QLineEdit::textChanged, [this, container]() {
      has_unsaved_changes_ = true;
      // Find which row this widget belongs to and update its bold state
      for (int row = 0; row < overrides_table_->rowCount(); ++row) {
        if (overrides_table_->cellWidget(row, 1) == container) {
          UpdateRowModifiedState(row);
          break;
        }
      }
    });
    layout->addWidget(line_edit);

    auto* browse_button = new QPushButton("Browse...");
    connect(browse_button, &QPushButton::clicked, [this, line_edit, var]() {
      QString path = QFileDialog::getExistingDirectory(
          this, SafeQString("Select Directory for " + var->name()),
          line_edit->text());
      if (!path.isEmpty()) {
        line_edit->setText(path);
      }
    });
    layout->addWidget(browse_button);

    return container;
  } else if (dynamic_cast<cvar::ConfigVar<int32_t>*>(var) ||
             dynamic_cast<cvar::ConfigVar<uint32_t>*>(var) ||
             dynamic_cast<cvar::ConfigVar<int64_t>*>(var) ||
             dynamic_cast<cvar::ConfigVar<uint64_t>*>(var)) {
    // Integer spin box
    auto* spinbox = new QSpinBox();
    spinbox->setRange(std::numeric_limits<int>::min(),
                      std::numeric_limits<int>::max());
    try {
      int value = std::stoi(current_value);
      spinbox->setValue(value);
    } catch (...) {
      spinbox->setValue(0);
    }
    connect(spinbox, QOverload<int>::of(&QSpinBox::valueChanged),
            [this, spinbox]() {
              has_unsaved_changes_ = true;
              // Find which row this widget belongs to and update its bold state
              for (int row = 0; row < overrides_table_->rowCount(); ++row) {
                if (overrides_table_->cellWidget(row, 1) == spinbox) {
                  UpdateRowModifiedState(row);
                  break;
                }
              }
            });
    return spinbox;
  } else {
    // Text input for all other types (string, double, etc.)
    auto* line_edit = new QLineEdit();
    line_edit->setText(SafeQString(current_value));
    connect(line_edit, &QLineEdit::textChanged, [this, line_edit]() {
      has_unsaved_changes_ = true;
      // Find which row this widget belongs to and update its bold state
      for (int row = 0; row < overrides_table_->rowCount(); ++row) {
        if (overrides_table_->cellWidget(row, 1) == line_edit) {
          UpdateRowModifiedState(row);
          break;
        }
      }
    });
    return line_edit;
  }
}

std::string GameConfigDialogQt::GetEditorValue(QWidget* editor) {
  if (auto* combo = qobject_cast<QComboBox*>(editor)) {
    return SafeStdString(combo->currentText());
  } else if (auto* spinbox = qobject_cast<QSpinBox*>(editor)) {
    return std::to_string(spinbox->value());
  } else if (auto* line_edit = qobject_cast<QLineEdit*>(editor)) {
    return SafeStdString(line_edit->text());
  } else if (auto* container = qobject_cast<QWidget*>(editor)) {
    // For path containers, find the QLineEdit child
    auto* line_edit = container->findChild<QLineEdit*>();
    if (line_edit) {
      return SafeStdString(line_edit->text());
    }
  }
  return "";
}

void GameConfigDialogQt::UpdateRowModifiedState(int row) {
  auto* name_item = overrides_table_->item(row, 0);
  if (!name_item) {
    return;
  }

  std::string var_name = SafeStdString(name_item->text());
  QWidget* value_widget = overrides_table_->cellWidget(row, 1);
  if (!value_widget) {
    return;
  }

  std::string current_value = GetEditorValue(value_widget);

  // Check if this value differs from the original
  auto original_it = original_overrides_.find(var_name);
  bool is_modified = (original_it == original_overrides_.end() ||
                      original_it->second != current_value);

  // Bold the name if modified
  QFont font = name_item->font();
  font.setBold(is_modified);
  name_item->setFont(font);
}

void GameConfigDialogQt::OnAddOverrideClicked() {
  // Create a simple dialog to select a cvar
  QDialog dialog(this);
  dialog.setWindowTitle("Add Config Override");
  dialog.setMinimumWidth(500);

  auto* layout = new QVBoxLayout(&dialog);

  auto* label =
      new QLabel("Select a configuration variable to override:", &dialog);
  layout->addWidget(label);

  auto* combo = new QComboBox(&dialog);

  // Populate with all available non-transient cvars
  if (cvar::ConfigVars) {
    std::vector<std::string> cvar_names;
    for (const auto& [name, var] : *cvar::ConfigVars) {
      if (!var->is_transient()) {
        cvar_names.push_back(name);
      }
    }
    std::sort(cvar_names.begin(), cvar_names.end());

    for (const auto& name : cvar_names) {
      auto* var = (*cvar::ConfigVars)[name];
      QString item_text =
          SafeQString(fmt::format("{} ({})", name, var->category()));
      combo->addItem(item_text, SafeQString(name));
    }
  }

  combo->setEditable(false);
  layout->addWidget(combo);

  auto* value_label = new QLabel("Value:", &dialog);
  layout->addWidget(value_label);

  // Create a container for the value editor that will be dynamically updated
  auto* value_container = new QWidget(&dialog);
  auto* value_container_layout = new QVBoxLayout(value_container);
  value_container_layout->setContentsMargins(0, 0, 0, 0);

  QWidget* current_editor = nullptr;
  cvar::IConfigVar* selected_var = nullptr;

  // Function to update the value editor based on selected cvar
  auto update_editor = [&, combo, value_container_layout, this]() {
    // Clear existing editor
    if (current_editor) {
      value_container_layout->removeWidget(current_editor);
      current_editor->deleteLater();
      current_editor = nullptr;
    }

    QString var_name = combo->currentData().toString();
    if (var_name.isEmpty() || !cvar::ConfigVars) {
      return;
    }

    selected_var = (*cvar::ConfigVars)[SafeStdString(var_name)];
    if (!selected_var) {
      return;
    }

    // Get current value from the cvar
    std::string current_value = selected_var->config_value();
    // Strip TOML quotes if present
    if (current_value.length() >= 2 && current_value.front() == '"' &&
        current_value.back() == '"') {
      current_value = current_value.substr(1, current_value.length() - 2);
    }

    // Create appropriate editor for this cvar type
    current_editor = CreateEditorForCvar(selected_var, current_value);
    value_container_layout->addWidget(current_editor);
  };

  // Update editor when combo box selection changes
  connect(combo, QOverload<int>::of(&QComboBox::currentIndexChanged),
          [update_editor]() { update_editor(); });

  // Initialize with first item
  if (combo->count() > 0) {
    update_editor();
  }

  layout->addWidget(value_container);

  auto* button_layout = new QHBoxLayout();
  auto* ok_button = new QPushButton("Add", &dialog);
  auto* cancel_button = new QPushButton("Cancel", &dialog);

  connect(ok_button, &QPushButton::clicked, &dialog, &QDialog::accept);
  connect(cancel_button, &QPushButton::clicked, &dialog, &QDialog::reject);

  button_layout->addStretch();
  button_layout->addWidget(ok_button);
  button_layout->addWidget(cancel_button);
  layout->addLayout(button_layout);

  if (dialog.exec() == QDialog::Accepted) {
    QString var_name = combo->currentData().toString();
    QString value;

    if (current_editor) {
      value = SafeQString(GetEditorValue(current_editor));
    }

    if (var_name.isEmpty()) {
      return;
    }

    // Check if this cvar is already in the table
    for (int row = 0; row < overrides_table_->rowCount(); ++row) {
      auto* item = overrides_table_->item(row, 0);
      if (item && item->text() == var_name) {
        QMessageBox::warning(
            this, "Duplicate Override",
            "This configuration variable is already being overridden.\n"
            "Edit the existing entry instead.");
        return;
      }
    }

    // Add new row
    CreateRow(SafeStdString(var_name), SafeStdString(value));

    has_unsaved_changes_ = true;
  }
}

void GameConfigDialogQt::OnSaveClicked() {
  SaveConfigOverrides();
  accept();
}

void GameConfigDialogQt::OnCancelClicked() {
  if (has_unsaved_changes_) {
    auto reply = QMessageBox::question(this, "Unsaved Changes",
                                       "You have unsaved changes. Are you sure "
                                       "you want to close without saving?",
                                       QMessageBox::Yes | QMessageBox::No);

    if (reply == QMessageBox::No) {
      return;
    }
  }

  reject();
}

void GameConfigDialogQt::OnUseRecommendedClicked() {
  LoadRecommendedSettings();
}

void GameConfigDialogQt::LoadRecommendedSettings() {
  auto settings_path = config::GetBundledDataPath("optimized_settings") /
                       (fmt::format("{:08X}", title_id_) + ".json");

  if (!std::filesystem::exists(settings_path)) {
    QMessageBox::information(this, "No Recommended Settings",
                             "No optimized settings are available for this "
                             "game in the community database.");
    return;
  }

  // Read the JSON file
  std::ifstream file(settings_path);
  if (!file.is_open()) {
    QMessageBox::warning(this, "Error Loading Settings",
                         "Failed to open recommended settings file.");
    return;
  }

  std::string json_content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
  file.close();

  // Parse JSON
  rapidjson::Document doc;
  rapidjson::ParseResult parse_result = doc.Parse(json_content.c_str());

  if (!parse_result) {
    QMessageBox::warning(
        this, "Error Parsing Settings",
        SafeQString(
            fmt::format("Failed to parse recommended settings: {} at offset {}",
                        rapidjson::GetParseError_En(parse_result.Code()),
                        parse_result.Offset())));
    return;
  }

  if (!doc.IsObject()) {
    QMessageBox::warning(this, "Error Parsing Settings",
                         "Recommended settings file has invalid format.");
    return;
  }

  // Track how many settings were applied
  int settings_applied = 0;

  // Iterate through categories (APU, GPU, HACKS, etc.)
  for (auto cat_it = doc.MemberBegin(); cat_it != doc.MemberEnd(); ++cat_it) {
    if (!cat_it->value.IsObject()) {
      continue;
    }

    // Iterate through settings within each category
    for (auto setting_it = cat_it->value.MemberBegin();
         setting_it != cat_it->value.MemberEnd(); ++setting_it) {
      std::string var_name = setting_it->name.GetString();
      std::string value = JsonValueToString(setting_it->value);

      // Apply CanaryCvarCompat translation
      std::string translated_var_name;
      std::string translated_value;
      CanaryCvarCompat(var_name, value, translated_var_name, translated_value);

      if (ApplyRecommendedSetting(translated_var_name, translated_value)) {
        settings_applied++;
      }
    }
  }

  if (settings_applied > 0) {
    has_unsaved_changes_ = true;
    QMessageBox::information(
        this, "Settings Loaded",
        SafeQString(
            fmt::format("Loaded {} recommended setting{} for this game.\n\n"
                        "Don't forget to save your changes!",
                        settings_applied, settings_applied == 1 ? "" : "s")));
  } else {
    QMessageBox::information(
        this, "No Settings Loaded",
        "No valid settings found in the recommended configuration.");
  }
}

bool GameConfigDialogQt::ApplyRecommendedSetting(const std::string& var_name,
                                                 const std::string& value) {
  // Check if this cvar exists in the system
  if (!cvar::ConfigVars ||
      (*cvar::ConfigVars).find(var_name) == (*cvar::ConfigVars).end()) {
    XELOGW("Unknown config variable in recommended settings: {}", var_name);
    return false;
  }

  auto* config_var = (*cvar::ConfigVars)[var_name];
  if (config_var->is_transient()) {
    return false;  // Skip transient cvars
  }

  // Check if this setting is already in the table
  for (int row = 0; row < overrides_table_->rowCount(); ++row) {
    auto* item = overrides_table_->item(row, 0);
    if (item && SafeStdString(item->text()) == var_name) {
      // Update the existing entry
      QWidget* value_widget = overrides_table_->cellWidget(row, 1);
      if (value_widget) {
        // Update the widget with the new value
        if (auto* combo = qobject_cast<QComboBox*>(value_widget)) {
          int index = combo->findText(SafeQString(value));
          if (index >= 0) {
            combo->setCurrentIndex(index);
          }
        } else if (auto* spinbox = qobject_cast<QSpinBox*>(value_widget)) {
          try {
            spinbox->setValue(std::stoi(value));
          } catch (...) {
            XELOGW("Failed to parse integer value for {}: {}", var_name, value);
          }
        } else if (auto* line_edit = qobject_cast<QLineEdit*>(value_widget)) {
          line_edit->setText(SafeQString(value));
        } else if (auto* container = qobject_cast<QWidget*>(value_widget)) {
          // For path containers, find the QLineEdit child
          auto* line_edit = container->findChild<QLineEdit*>();
          if (line_edit) {
            line_edit->setText(SafeQString(value));
          }
        }
      }
      return true;  // Done updating existing entry
    }
  }

  // Add new row since this setting doesn't exist yet
  CreateRow(var_name, value);

  return true;
}

void GameConfigDialogQt::CreateRow(const std::string& var_name,
                                   const std::string& value) {
  int row = overrides_table_->rowCount();
  overrides_table_->insertRow(row);

  // Column 0: Variable name (non-editable text)
  auto* name_item = new QTableWidgetItem(SafeQString(var_name));
  name_item->setFlags(name_item->flags() & ~Qt::ItemIsEditable);
  overrides_table_->setItem(row, 0, name_item);

  // Column 1: Value editor widget based on type
  cvar::IConfigVar* config_var = nullptr;
  if (cvar::ConfigVars &&
      (*cvar::ConfigVars).find(var_name) != (*cvar::ConfigVars).end()) {
    config_var = (*cvar::ConfigVars)[var_name];
  }
  QWidget* value_editor = CreateEditorForCvar(config_var, value);
  overrides_table_->setCellWidget(row, 1, value_editor);

  // Column 2: Delete button (centered)
  auto* delete_button = new QPushButton("🗑", this);
  delete_button->setMaximumWidth(30);
  delete_button->setToolTip("Remove this override");
  auto* button_container = new QWidget(this);
  auto* button_layout = new QHBoxLayout(button_container);
  button_layout->addWidget(delete_button);
  button_layout->setAlignment(Qt::AlignCenter);
  button_layout->setContentsMargins(0, 0, 0, 0);
  overrides_table_->setCellWidget(row, 2, button_container);

  // Connect after setting the cell widget so we can find the row dynamically
  connect(delete_button, &QPushButton::clicked, [this, button_container]() {
    // Find which row this button is in
    for (int i = 0; i < overrides_table_->rowCount(); ++i) {
      if (overrides_table_->cellWidget(i, 2) == button_container) {
        overrides_table_->removeRow(i);
        has_unsaved_changes_ = true;
        break;
      }
    }
  });
}

}  // namespace app
}  // namespace xe
