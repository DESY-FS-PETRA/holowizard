import json
import os
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QTextEdit, QPushButton, QMessageBox, QHBoxLayout
)

class AdvancedJsonEditor(QDialog):
    """
    A dialog window to allow users to view and edit a JSON configuration file.

    This editor opens the config as plain text, allows modifications,
    and validates the JSON before saving.
    """

    def __init__(self, config_path, parent=None):
        """
        Initialize the editor dialog.

        Parameters:
            config_path (str): Path to the JSON configuration file.
            parent (QWidget): Optional parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("Advanced Settings – Edit Config JSON")
        self.resize(700, 500)
        self.config_path = config_path

        # Setup layout
        layout = QVBoxLayout(self)

        # Instruction label
        self.label = QLabel("Edit configuration (JSON):")
        layout.addWidget(self.label)

        # JSON text editor
        self.editor = QTextEdit(self)
        layout.addWidget(self.editor)

        # Load content from file
        self.load_json()

        # Save/Cancel button row
        button_layout = QHBoxLayout()
        self.save_button = QPushButton("Save")
        self.cancel_button = QPushButton("Cancel")
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        # Connect signals
        self.save_button.clicked.connect(self.save_json)
        self.cancel_button.clicked.connect(self.reject)

    def load_json(self):
        """
        Load the config file into the editor.

        If the file doesn't exist or is invalid, show an error and close the dialog.
        """
        if not os.path.exists(self.config_path):
            QMessageBox.critical(self, "Error", "Config file not found.")
            self.reject()
            return

        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            pretty_json = json.dumps(config_data, indent=4)
            self.editor.setPlainText(pretty_json)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load JSON:\n{e}")
            self.reject()

    def save_json(self):
        """
        Validate and save the JSON content to file.

        If the content is invalid, a warning is shown instead.
        """
        try:
            new_json_text = self.editor.toPlainText()
            new_config = json.loads(new_json_text)  # Validate formatting
            with open(self.config_path, 'w') as f:
                json.dump(new_config, f, indent=4)
            QMessageBox.information(self, "Saved", "Configuration saved successfully.")
            self.accept()
        except json.JSONDecodeError as e:
            QMessageBox.warning(self, "Invalid JSON", f"Please fix JSON formatting:\n{e}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save JSON:\n{e}")
