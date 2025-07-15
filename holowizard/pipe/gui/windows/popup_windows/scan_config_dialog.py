from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QFormLayout, QLineEdit, QComboBox, QCheckBox, QPushButton, QHBoxLayout, QLabel
from PyQt5.QtGui import QDoubleValidator

class ScanConfigDialog(QDialog):
    """
    A dialog window to configure parameters for scan processing.

    Parameters:
        scan_name (str): The name of the scan (displayed but not editable).
        parent (QWidget): The parent widget.
    """
    def __init__(self, scan_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scan Processing Configuration")

        self.scan_name = scan_name
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        form_layout = QFormLayout()

        # Scan name (read-only)
        self.scan_name_display = QLabel(self.scan_name)
        form_layout.addRow("Sample Name:", self.scan_name_display)

        # Energy input
        self.energy_input = QLineEdit()
        self.energy_input.setValidator(QDoubleValidator())
        form_layout.addRow("Energy (keV):", self.energy_input)

        # Phase config
        self.phase_config_input = QComboBox()
        self.phase_config_input.addItems(["Mg Wire", "Tooth", "Spyder Hair", 'Cactus Needle'])
        form_layout.addRow("Regularization:", self.phase_config_input)

        # Use P05 optics (checkbox)
        self.use_p05_checkbox = QCheckBox("Use P05 Optics")
        self.use_p05_checkbox.toggled.connect(self.toggle_z_fields)
        form_layout.addRow(self.use_p05_checkbox)

        # Holder input
        self.holder_input = QComboBox()
        self.holder_input.addItems(["1 (80 mm)", "2 (195 mm)", "3 (220 mm)", '4 (250 mm)'])
        form_layout.addRow("Holder:", self.holder_input)

        # z01/z02 inputs
        self.z01_input = QLineEdit()
        self.z01_input.setValidator(QDoubleValidator())
        self.z02_input = QLineEdit()
        self.z02_input.setValidator(QDoubleValidator())
        form_layout.addRow(u"Z01 (µm):", self.z01_input)
        form_layout.addRow(u"Z02 (µm):", self.z02_input)

        layout.addLayout(form_layout)

        # Buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

        self.toggle_z_fields()  # Initialize the state

    def toggle_z_fields(self):
        """
        Enable or disable z01/z02 or holder fields depending on optics selection.
        """
        use_p05 = self.use_p05_checkbox.isChecked()
        self.z01_input.setEnabled(not use_p05)
        self.z02_input.setEnabled(not use_p05)
        self.holder_input.setEnabled(use_p05)

    def get_values(self):
        """
        Retrieve input values from the dialog.

        Returns:
            dict: Dictionary containing the configuration values.
        """
        return {
            "energy": float(self.energy_input.text()),
            "z01": float(self.z01_input.text()) if self.z01_input.isEnabled() else None,
            "z02": float(self.z02_input.text()) if self.z02_input.isEnabled() else None,
            "holder": float(self.holder_input.currentText().split('(')[1].split(' ')[
                                0]) if self.holder_input.isEnabled() else None,
            "use_p05": self.use_p05_checkbox.isChecked(),
            "phase_config": self.phase_config_input.currentText()
        }
