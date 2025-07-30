# Authors: Silja Flenner, André Lopes Marinho
import configparser
import datetime
import os
import subprocess
import sys
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (
    QMainWindow,
    QTableWidgetItem,
    QHBoxLayout,
    QVBoxLayout,
    QTableWidget,
    QSlider,
    QLineEdit,
    QPushButton,
    QLabel,
    QTabWidget,
    QHeaderView,
    QWidget,
    QTextEdit,
    QFormLayout,
)
from PyQt5.QtCore import QTimer, Qt

from holowizard.beamtime import bt_utils as bt
from gui.gui_utils.find_focus_thread import FindFocusThread
from gui.gui_utils.logging_editor import QTextEditLogger
from gui.gui_utils.phase_retrieval_thread import PhaseRetrievalThread
from gui.gui_utils.widgets_utils import create_divider
from holowizard.beamtime.bt_utils import get_user_accessible_years
from gui.windows.popup_windows.scan_config_dialog import ScanConfigDialog

from silx.gui.plot import Plot2D


from PyQt5.QtWidgets import (
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QHBoxLayout,
    QPushButton,
    QFormLayout,
)


class MainWindow(QMainWindow):
    def __init__(self):
        """
        Initialize the main application window and set up the UI components.
        """
        super().__init__()

        # Initialization flag for setupUi
        self.ui_initialized = False

        # Save original stdout and stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        # Setup phase retrieval tab properties
        self.phase_retrieval_thread = None
        self.refresh_timer = QTimer()
        self.refresh_timer.setInterval(500)

        self.img_num = 1
        self.current_bt = None
        self.current_scan = None
        self.current_year = None
        self.year = datetime.datetime.now().year
        self.bold_font = "font-weight: bold;"

        # Path to the default.ini file
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.config_path = os.path.join(root_dir, "default.ini")

        # Initialize configparser and load the .ini file
        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        self.config.read(self.config_path)

        self.slice_folder = "/path/to/slices"  # Update with the correct path
        self.phase_folder = "/path/to/phase_retrieved"
        self.stack_folder = "/path/to/stack"
        self.current_view = "slice"  # Default view
        self.current_max_image = 0  # Will be updated dynamically

        # Only initialize UI once
        if not self.ui_initialized:
            self.setupUi()
            self.ui_initialized = True

        # Connect UI elements to their functions
        self.setup_connections()

    def setupUi(self):
        """
        Set up the user interface, including tabs and widgets for different application features.
        """
        self.setWindowTitle("HoloPipe")
        self.resize(1200, 800)

        # Create a central widget and main layout
        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        self.tabs = QTabWidget()

        # Setup each tab
        self.setup_home_tab()
        self.setup_beamtime_viewer_tab()
        self.setup_phase_retrieval_tab()
        self.setup_tomo_reco_tab()
        self.setup_holopipe_tab()

        # Final setup for main layout
        main_layout.addWidget(self.tabs)

    def setup_connections(self):
        """
        Connect UI components to their respective event handlers.
        """
        self.comboBox_year.currentIndexChanged.connect(self.load_beamtimes)
        self.tableWidget.cellClicked.connect(self.select_beamtime)
        self.tableWidget_scans.cellClicked.connect(self.select_scan)
        self.button_open.clicked.connect(self.click_open)
        self.button_search.clicked.connect(self.search)
        self.slider_num.valueChanged.connect(self.slider_image_num)
        self.button_prev.clicked.connect(self.prev_image)
        self.button_next.clicked.connect(self.next_image)
        self.button_add_scan_to_processing.clicked.connect(self.add_scan_to_processing)
        self.button_start_reconstruction.clicked.connect(self.start_phase_retrieval)
        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.refresh_timer.timeout.connect(self.update_canvas_during_reconstruction)
        self.button_advanced_settings.clicked.connect(self.open_advanced_settings)
        self.button_prev_phase_image.clicked.connect(self.prev_image)
        self.button_next_phase_image.clicked.connect(self.next_image)
        self.button_start_reconstruction.clicked.connect(self.start_phase_retrieval)
        self.button_find_focus.clicked.connect(self.start_find_focus)
        self.button_stop_reconstruction.clicked.connect(self.stop_reconstruction)
        self.image_num_input.returnPressed.connect(self.on_image_num_entered)
        self.clear_log_button.clicked.connect(self.clear_logging)
        self.clear_canvas_button.clicked.connect(self.clear_iteration_canvas)
        self.button_phase_viewer.clicked.connect(self.open_phase_viewer_popup)
        self.button_search_rotation_center.clicked.connect(self.open_rotation_center_search)
        self.button_reconstruct_stack.clicked.connect(self.reconstruct_stack)
        self.comboBox_year.currentIndexChanged.connect(self.on_year_changed)

        # Disable the slider by default
        self.slider_num.setEnabled(False)

    def setup_home_tab(self):
        """
        Set up the home tab with software description and functionalities overview.
        """
        self.about_tab = QtWidgets.QWidget()
        about_layout = QVBoxLayout(self.about_tab)
        self.about_text = QTextEdit(self)
        self.about_text.setReadOnly(True)
        self.about_text.setHtml(
            """
            <h2>Welcome to HoloPipe</h2>
            <p>This software is designed to assist with phase retrieval and beamtime management for scientific imaging. 
            Below is an overview of the main functionalities:</p>

            <h3>Beamtime Viewer</h3>
            <p>The Beamtime Viewer tab allows you to view and select beamtimes and scans, providing search and 
            selection functionality to easily navigate data.</p>

            <h3>Phase Retrieval</h3>
            <p>The Phase Retrieval tab allows you to start and monitor the phase retrieval process. You can adjust 
            advanced settings and view iterative reconstruction results in real-time.</p>

            <p>Please explore each tab for specific functionalities. For more detailed information, consult the user 
            manual or contact support.</p>
            """
        )
        about_layout.addWidget(self.about_text)
        self.tabs.addTab(self.about_tab, "Home")

    def setup_beamtime_viewer_tab(self):
        """
        Set up the Beamtime Viewer tab to display available beamtimes and scans for selection.
        """

        # Setting up tab
        self.beamtime_viewer_tab = QtWidgets.QWidget()

        # Beamtime viewer layout
        beamtime_tab_layout = QHBoxLayout(self.beamtime_viewer_tab)

        # Left side for year selection, beamtime table, and controls
        left_beamtime_layout = QVBoxLayout()

        # Select year dropbox
        self.beamtime_year_label = QLabel("Select the beamtime year")
        self.beamtime_year_label.setStyleSheet(self.bold_font)
        self.comboBox_year = QtWidgets.QComboBox(self)
        self.comboBox_year.clear()
        years = get_user_accessible_years()
        if years:
            self.comboBox_year.addItems(["Select year..."])
            self.comboBox_year.addItems(years)
        else:
            self.comboBox_year.addItem("No access to beamtime data")
            self.comboBox_year.setEnabled(False)
        left_beamtime_layout.addWidget(self.beamtime_year_label)
        left_beamtime_layout.addWidget(self.comboBox_year)

        # Beamtime table
        self.tableWidget = QTableWidget(self)
        self.beamtimes_label = QLabel("Beamtimes")
        self.beamtimes_label.setStyleSheet(self.bold_font)
        left_beamtime_layout.addWidget(self.beamtimes_label)
        left_beamtime_layout.addWidget(self.tableWidget)

        # Scan table
        self.tableWidget_scans = QTableWidget(self)
        self.scan_list_label = QLabel("Scan list")
        self.scan_list_label.setStyleSheet(self.bold_font)
        left_beamtime_layout.addWidget(self.scan_list_label)
        left_beamtime_layout.addWidget(self.tableWidget_scans)
        self.tableWidget_scans.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        search_layout = QHBoxLayout()
        self.edit_search = QLineEdit(self)
        self.edit_search.setPlaceholderText("Search for a scan...")
        self.button_search = QPushButton("Search", self)
        search_layout.addWidget(self.edit_search)
        search_layout.addWidget(self.button_search)
        left_beamtime_layout.addLayout(search_layout)

        button_layout = QHBoxLayout()
        self.button_open = QPushButton("Open Beamtime Folder", self)
        self.button_add_scan_to_processing = QPushButton("Select scan for processing", self)
        button_layout.addWidget(self.button_add_scan_to_processing)
        button_layout.addWidget(self.button_open)
        left_beamtime_layout.addLayout(button_layout)

        beamtime_tab_layout.addLayout(left_beamtime_layout)

        # Right side for image viewer and controls
        right_beamtime_layout = QVBoxLayout()
        self.imv = Plot2D()
        self.imv.setKeepDataAspectRatio(True)
        right_beamtime_layout.addWidget(self.imv)

        slider_layout = QHBoxLayout()
        self.button_prev = QPushButton("<", self)
        self.slider_num = QSlider(Qt.Horizontal, self)
        self.slider_num.setEnabled(False)
        self.button_next = QPushButton(">", self)
        slider_layout.addWidget(self.button_prev)
        slider_layout.addWidget(self.slider_num)
        slider_layout.addWidget(self.button_next)
        right_beamtime_layout.addLayout(slider_layout)

        self.image_info = QLabel(self)
        right_beamtime_layout.addWidget(self.image_info)

        beamtime_tab_layout.addLayout(right_beamtime_layout)
        self.beamtime_viewer_tab.setLayout(beamtime_tab_layout)
        self.tabs.addTab(self.beamtime_viewer_tab, "Beamtime Viewer")

    def setup_phase_retrieval_tab(self):
        """
        Set up the Phase Retrieval tab to configure and run phase retrieval or focus finding.
        """
        self.phase_retrieval_tab = QWidget()
        phase_layout = QHBoxLayout(self.phase_retrieval_tab)

        # Left side for metadata display and controls
        left_phase_layout = QVBoxLayout()

        # Matplotlib canvas for image display with color bar
        self.fig_plot = Figure()
        self.canvas_plot = FigureCanvas(self.fig_plot)
        left_phase_layout.addWidget(self.canvas_plot)

        # Add a slider and navigation buttons to scroll through images
        image_slider_layout = QHBoxLayout()
        self.button_prev_phase_image = QPushButton("<")
        self.button_next_phase_image = QPushButton(">")
        self.image_num_input = QLineEdit()
        self.image_num_input.setFixedWidth(100)
        self.image_num_input.setAlignment(Qt.AlignCenter)
        self.image_num_input.setText(str(self.img_num))
        image_slider_layout.addWidget(self.button_prev_phase_image)
        image_slider_layout.addWidget(self.image_num_input)
        image_slider_layout.addWidget(self.button_next_phase_image)

        left_phase_layout.addLayout(image_slider_layout)

        # Metadata labels for scan and reconstruction status
        self.info_label = QLabel("Info")
        self.info_label.setStyleSheet(self.bold_font)
        self.label_scan_name = QLabel("Scan: ")
        self.reconstruction_status_label = QLabel("Status: Ready")
        left_phase_layout.addWidget(create_divider())
        left_phase_layout.addWidget(self.info_label)
        left_phase_layout.addWidget(self.label_scan_name)
        left_phase_layout.addWidget(self.reconstruction_status_label)

        # Parameters input fields
        self.param_label = QLabel("Parameters")
        self.param_label.setStyleSheet(self.bold_font)
        param_layout = QFormLayout()
        left_phase_layout.addWidget(create_divider())
        left_phase_layout.addWidget(self.param_label)
        left_phase_layout.addLayout(param_layout)

        # Set Parameters button to save inputs to ini file
        button_row_layout = QHBoxLayout()
        self.button_advanced_settings = QPushButton("Advanced Settings", self)
        button_row_layout.addWidget(self.button_advanced_settings)
        left_phase_layout.addLayout(button_row_layout)

        # Control buttons for phase retrieval, focus finding, and stopping
        left_phase_layout.addWidget(create_divider())
        self.processing_label = QLabel("Processing")
        self.processing_label.setStyleSheet(self.bold_font)
        self.button_start_reconstruction = QPushButton("Start Phase Retrieval")
        self.button_find_focus = QPushButton("Find Focus")
        self.button_stop_reconstruction = QPushButton("Stop")
        left_phase_layout.addWidget(self.processing_label)
        left_phase_layout.addWidget(self.button_start_reconstruction)
        left_phase_layout.addWidget(self.button_find_focus)
        left_phase_layout.addWidget(self.button_stop_reconstruction)
        self.button_stop_reconstruction.setEnabled(False)  # Disable the stop button initially

        # Right side for reconstruction viewer and log display
        right_phase_layout = QVBoxLayout()

        # Label and Clear button for the Iteration viewer
        self.iteration_label = QLabel("Iteration Viewer")
        self.iteration_label.setStyleSheet(self.bold_font)
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        # Adjust Clear Iteration Canvas Button size and alignment
        self.clear_canvas_button = QPushButton("Clear viewer")
        self.clear_canvas_button.setFixedWidth(150)
        clear_canvas_button_layout = QHBoxLayout()
        spacer = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        clear_canvas_button_layout.addItem(spacer)
        clear_canvas_button_layout.addWidget(self.clear_canvas_button)
        right_phase_layout.addWidget(self.iteration_label)
        right_phase_layout.addWidget(self.canvas)
        right_phase_layout.addLayout(clear_canvas_button_layout)

        # Label and Clear button for the Logging
        self.log_label = QLabel("Logging")
        self.log_label.setStyleSheet(self.bold_font)
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setStyleSheet("font-size: 12px;")
        self.log_display.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.log_display.setFixedHeight(250)
        self.clear_log_button = QPushButton("Clear log")
        self.clear_log_button.setFixedWidth(150)
        clear_log_button_layout = QHBoxLayout()
        spacer = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        clear_log_button_layout.addItem(spacer)
        clear_log_button_layout.addWidget(self.clear_log_button)
        right_phase_layout.addWidget(self.log_label)
        right_phase_layout.addWidget(self.log_display)
        right_phase_layout.addLayout(clear_log_button_layout)

        # Combine layouts
        phase_layout.addLayout(left_phase_layout, 2)
        phase_layout.addLayout(right_phase_layout, 5)
        self.phase_retrieval_tab.setLayout(phase_layout)
        self.tabs.addTab(self.phase_retrieval_tab, "Phase Retrieval")

    def setup_tomo_reco_tab(self):
        """
        Set up the Tomographic Reconstruction tab for reconstructing slices or stacks.
        """

        self.tomo_reco_tab = QWidget()
        tomo_layout = QHBoxLayout(self.tomo_reco_tab)

        # === Left Panel ===
        left_tomo_layout = QVBoxLayout()

        # Info Section
        left_tomo_layout.addWidget(create_divider())
        info_label = QLabel("Info")
        info_label.setStyleSheet(self.bold_font)
        left_tomo_layout.addWidget(info_label)
        self.label_tomo_scan_name = QLabel("Scan: ")
        self.label_tomo_status = QLabel("Status: Ready")
        left_tomo_layout.addWidget(self.label_tomo_scan_name)
        left_tomo_layout.addWidget(self.label_tomo_status)

        # Pre-processing Section
        left_tomo_layout.addWidget(create_divider())
        pre_processing_label = QLabel("Pre-Processing")
        pre_processing_label.setStyleSheet(self.bold_font)
        left_tomo_layout.addWidget(pre_processing_label)
        self.button_phase_viewer = QPushButton("View Phase Retrieved Images")
        self.button_show_sinogram = QPushButton("Show Sinogram")
        left_tomo_layout.addWidget(self.button_phase_viewer)
        left_tomo_layout.addWidget(self.button_show_sinogram)

        # Parameters Section
        left_tomo_layout.addWidget(create_divider())
        parameters_label = QLabel("Parameters")
        parameters_label.setStyleSheet(self.bold_font)
        left_tomo_layout.addWidget(parameters_label)
        param_layout = QFormLayout()
        self.step_size_input = QLineEdit()
        self.delta_input = QLineEdit()
        self.rec_slice_input = QLineEdit()
        self.algorithm_dropdown = QtWidgets.QComboBox()
        self.algorithm_dropdown.addItems(["FBP", "SART", "ART"])  # Example algorithms
        param_layout.addRow("Step Size:", self.step_size_input)
        param_layout.addRow("Delta:", self.delta_input)
        param_layout.addRow("Reconstruction Slice:", self.rec_slice_input)
        param_layout.addRow("Algorithm:", self.algorithm_dropdown)
        left_tomo_layout.addLayout(param_layout)

        # Processing Buttons
        left_tomo_layout.addWidget(create_divider())
        self.button_search_rotation_center = QPushButton("Search Rotation Center")
        self.button_reconstruct_stack = QPushButton("Reconstruct Stack")
        self.button_stop_tomo = QPushButton("Stop")
        left_tomo_layout.addWidget(self.button_search_rotation_center)
        left_tomo_layout.addWidget(self.button_reconstruct_stack)
        left_tomo_layout.addWidget(self.button_stop_tomo)

        # === Right Panel ===
        right_tomo_layout = QVBoxLayout()

        # Silx Viewer for the final reconstructed stack
        self.stack_image_viewer = Plot2D()
        self.stack_image_viewer.setKeepDataAspectRatio(True)
        right_tomo_layout.addWidget(self.stack_image_viewer)

        # Combine left and right panels
        tomo_layout.addLayout(left_tomo_layout, 2)
        tomo_layout.addLayout(right_tomo_layout, 5)
        self.tomo_reco_tab.setLayout(tomo_layout)
        self.tabs.addTab(self.tomo_reco_tab, "Tomographic Reconstruction")

    # Popup for rotation center search
    def open_rotation_center_search(self):
        """
        Open a popup to allow searching for the best rotation center.
        """
        self.rotation_center_popup = QtWidgets.QWidget()
        self.rotation_center_popup.setWindowTitle("Rotation Center Search")
        layout = QVBoxLayout(self.rotation_center_popup)

        # Silx viewer for rotation center stack
        self.rotation_viewer = Plot2D()
        self.rotation_viewer.setKeepDataAspectRatio(True)
        layout.addWidget(self.rotation_viewer)

        # Slider for scrolling rotation center stack
        slider_layout = QHBoxLayout()
        self.rotation_slider = QSlider(Qt.Horizontal)
        slider_layout.addWidget(self.rotation_slider)
        layout.addLayout(slider_layout)

        # Button for reconstructing with selected center
        self.button_reconstruct_with_center = QPushButton("Reconstruct with Selected Center")
        layout.addWidget(self.button_reconstruct_with_center)

        self.rotation_center_popup.setLayout(layout)
        self.rotation_center_popup.resize(800, 600)
        self.rotation_center_popup.show()

        # Connect reconstruct button
        self.button_reconstruct_with_center.clicked.connect(self.reconstruct_with_selected_center)

    # Functions for reconstruction
    def reconstruct_stack(self):
        """
        Perform tomographic stack reconstruction and display in Silx viewer.
        """
        self.label_tomo_status.setText("Reconstructing stack...")
        # Perform stack reconstruction (dummy example)
        stack = np.random.rand(100, 256, 256)  # Replace with actual reconstruction
        self.load_tomographic_stack(stack)
        self.label_tomo_status.setText("Stack reconstruction complete.")

    def reconstruct_with_selected_center(self):
        """
        Reconstruct a slice using the selected rotation center.
        """
        self.label_tomo_status.setText("Reconstructing slice with selected center...")
        # Perform slice reconstruction (dummy example)
        selected_center = self.rotation_slider.value()
        print(f"Reconstructing slice with center: {selected_center}")
        self.rotation_center_popup.close()

    def setup_holopipe_tab(self):
        """
        Set up the HoloPipe tab as a placeholder for future functionality.
        """
        self.holopipe_tab = QtWidgets.QWidget()
        holopipe_layout = QVBoxLayout(self.holopipe_tab)
        self.holopipe_holder_text = QTextEdit(self)
        self.holopipe_holder_text.setReadOnly(True)
        self.holopipe_holder_text.setHtml("<h2>Holder for HoloPipe</h2>")
        holopipe_layout.addWidget(self.holopipe_holder_text)
        self.tabs.addTab(self.holopipe_tab, "HoloPipe")

    def on_tab_changed(self, index):
        """
        Handle actions specific to each tab when it is activated.
        """
        if index == self.tabs.indexOf(self.beamtime_viewer_tab):
            if self.current_scan:
                self.display_silx_image()

        elif index == self.tabs.indexOf(self.phase_retrieval_tab):
            if self.current_scan:
                self.update_placeholder_image(self.current_scan.load_image(self.img_num))
            self.canvas.draw_idle()

    def display_silx_image(self):
        """
        Display the currently selected image in the Silx viewer.
        """
        if self.current_scan and self.img_num <= self.current_scan.num_img:
            self.process_img_for_display(self.img_num)

    def load_beamtimes(self):
        """
        Load beamtimes based on the selected year and populate the table.
        """
        self.year = str(self.comboBox_year.currentText())

        # Clear the scan list table when the year is changed
        self.tableWidget_scans.clearContents()
        self.tableWidget_scans.setRowCount(0)  # Ensure scan table is empty

        # Clear the selection in the beamtime table
        self.tableWidget.clearSelection()

        self.bt_list = bt.load_beamtime_list(self.year)

        display_keys = ["beamtimeId", "eventStart", "title", "pi", "leader"]
        self.tableWidget.setRowCount(len(self.bt_list))
        self.tableWidget.setColumnCount(len(display_keys))

        for row, beamtime in enumerate(self.bt_list):
            for col, key in enumerate(display_keys):
                item = beamtime.meta_dict.get(key, "")
                items = item["lastname"] if isinstance(item, dict) else str(item)
                self.tableWidget.setHorizontalHeaderItem(col, QTableWidgetItem(key))
                self.tableWidget.setItem(row, col, QTableWidgetItem(items))

    def select_beamtime(self, row):
        """
        Select a beamtime and display associated scans in the scan table.
        """
        self.current_bt = self.bt_list[row]
        self.scanlist = self.current_bt.scans_tab
        self.tableWidget_scans.setRowCount(len(self.scanlist))
        self.tableWidget_scans.setColumnCount(1)
        self.tableWidget_scans.setHorizontalHeaderItem(0, QTableWidgetItem("scan"))

        for row, scan in enumerate(self.scanlist):
            self.tableWidget_scans.setItem(row, 0, QTableWidgetItem(scan.name))

        self.slider_num.setEnabled(False)

    def search(self):
        """
        Search for a specific scan across all beamtimes and display the results.
        """
        # Check if beamtime list exists and is non-empty
        if not hasattr(self, "bt_list") or not self.bt_list:
            QtWidgets.QMessageBox.warning(
                self,
                "No Beamtime Loaded",
                "Please select a year and load beamtimes before searching for scans.",
            )
            return

        self.search_item = self.edit_search.text().strip()
        if not self.search_item:
            QtWidgets.QMessageBox.information(self, "Empty Search", "Please enter a scan name to search for.")
            return

        self.scanlist = [result for bt in self.bt_list for result in bt.search_in_bt(self.search_item)]

        self.tableWidget_scans.setRowCount(len(self.scanlist))
        self.tableWidget_scans.setColumnCount(1)
        self.tableWidget_scans.setHorizontalHeaderItem(0, QTableWidgetItem("Scan"))

        for row, scan in enumerate(self.scanlist):
            self.tableWidget_scans.setItem(row, 0, QTableWidgetItem(scan.name))

        if not self.scanlist:
            QtWidgets.QMessageBox.information(self, "No Results", f"No scans found for: {self.search_item}")

    def select_scan(self, row):
        """
        Select a scan and display it in the Silx image viewer.
        Update default.ini with the selected beamtime, year, and scan.
        """
        self.current_scan = self.scanlist[row]
        self.img_num = 1
        self.process_img_for_display(self.img_num)
        self.slider_num.setRange(1, self.current_scan.num_img)
        self.slider_num.setEnabled(True)
        self.update_image_info()

        # Update the Scan info in the Tomographic Reconstruction Tab
        self.label_tomo_scan_name.setText(f"Scan: {self.current_scan.name}")

    # def on_img_number_changed(self):
    #     """
    #     Update the img_num variable in the default.ini file.
    #     """
    #     try:
    #         update_ini_section(
    #             self.config_path,
    #             section='single_phase_retrieval',
    #             updates={
    #                 'img_index': self.img_num
    #             }
    #         )
    #
    #     except Exception as e:
    #         QtWidgets.QMessageBox.warning(
    #             self, "Update Failed",
    #             f"An error occurred while updating default.ini: {e}"
    #         )

    def on_year_changed(self, index):
        """
        Update the current_year variable when the year dropdown changes.
        """

        if index == 0:
            return
        self.current_year = self.comboBox_year.currentText()

    def slider_image_num(self):
        """
        Update the displayed image when the slider is moved.
        """
        if self.current_scan:
            self.img_num = int(self.slider_num.value())
            self.process_img_for_display(self.img_num)
            self.update_image_info()

    def update_image_info(self):
        """
        Update the label displaying the current image number.
        """
        if self.current_scan:
            self.image_info.setText(
                f"Scan: {self.current_scan.name} | Image {self.img_num} of {self.current_scan.num_img}"
            )
            self.image_num_input.setText(str(self.img_num))

    def prev_image(self):
        """
        Navigate to the previous image in the scan.
        """
        if self.current_scan and self.img_num > 1:
            self.img_num -= 1
            self.slider_num.setValue(self.img_num)
            self.update_image_info()

            # Update image based on the active tab
            if self.tabs.currentWidget() == self.phase_retrieval_tab:
                self.update_placeholder_image(self.current_scan.load_image(self.img_num))
                self.image_num_input.setText(str(self.img_num))
            else:
                self.display_silx_image()

    def next_image(self):
        """
        Navigate to the next image in the scan.
        """
        if self.current_scan and self.img_num < self.current_scan.num_img:
            self.img_num += 1
            self.slider_num.setValue(self.img_num)
            self.update_image_info()

            # Update image based on the active tab
            if self.tabs.currentWidget() == self.phase_retrieval_tab:
                self.update_placeholder_image(self.current_scan.load_image(self.img_num))
                self.image_num_input.setText(str(self.img_num))
            else:
                self.display_silx_image()

    def click_open(self):
        """
        Open the current beamtime directory in the file explorer.
        """
        if self.current_bt is not None:
            path = f"/asap3/petra3/gpfs/p05/{self.year}/data/{self.current_bt.beamtime}"
            subprocess.Popen(["xdg-open", path])
        else:
            QtWidgets.QMessageBox.warning(self, "Warning!", "Please select a beamtime!")

    def add_scan_to_processing(self):
        """
        When the user clicks 'Select scan for processing', generate a config file.
        """

        if not self.current_bt or not self.current_scan:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select a beamtime and a scan first.")
            return

        dialog = ScanConfigDialog(scan_name=self.current_scan.name, parent=self)
        if dialog.exec_():
            config_data = dialog.get_values()

            try:
                from holowizard.builders.project_config_builder import (
                    ProjectConfigBuilder,
                )

                # Holder parser
                def parse_holder(holder_text: str) -> float:
                    try:
                        return float(holder_text.split("(")[1].split(" ")[0])
                    except Exception:
                        return 220.0  # default fallback

                # Map phase label to internal config name
                phase_type_map = {
                    "Mg Wire": "wire",
                    "Cactus Needle": "cactus",
                    "Tooth": "tooth",
                    "Spyder Hair": "spider",
                }

                # Create a P05Scan object using selected scan
                scan = self.current_scan
                scan.energy = config_data["energy"]
                scan.holder_length = parse_holder(config_data["holder"]) if config_data["use_p05"] else 220.0

                builder = ProjectConfigBuilder(scan)
                builder.generate_default_config()

                # Apply user config
                builder.config["default_options"]["default_phase_params"] = phase_type_map[config_data["phase_config"]]
                builder.config["default_options"]["p05_geometry"] = config_data["use_p05"]
                builder.config["project"]["energy"] = config_data["energy"]
                builder.config["project"]["holder"] = scan.holder_length
                builder.config["project"]["optics_qp"] = config_data["use_p05"]

                if not config_data["use_p05"]:
                    builder.config["z_params"]["z01"] = config_data["z01"]
                    builder.config["z_params"]["z02"] = config_data["z02"]

                path = builder.save_config()

                QtWidgets.QMessageBox.information(self, "Success", f"Configuration saved to:\n{path}")

            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to create config:\n{e}")

    def save_config(self):
        """
        Save the modified values from the UI back to the default.ini file.
        """
        # Iterate over form_widgets and get the updated values
        for (section, key), widget in self.form_widgets.items():
            value = widget.text()  # Get the text from QLineEdit
            self.config.set(section, key, value)

        # Write the updated configuration to the ini file
        with open(self.config_path, "w") as configfile:
            self.config.write(configfile)

        # Notify the user that the save was successful
        QtWidgets.QMessageBox.information(self, "Success", "Configuration saved!")

    def update_placeholder_image(self, img):
        """
        Update the placeholder canvas with the selected image, centered, and display scan/image names.
        """
        self.fig_plot.clear()

        # Add image with color bar, ensuring it is centered
        ax = self.fig_plot.add_subplot(111)
        img_rotated = np.rot90(img)
        vmin, vmax = np.percentile(img_rotated, (1, 99))
        cax = ax.imshow(img_rotated, cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)
        self.fig_plot.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

        # Ensure plot is centered within canvas
        ax.axis("off")
        ax.set_aspect("auto")  # Automatically adjust aspect ratio
        self.fig_plot.tight_layout(pad=0)  # Remove extra padding around the plot

        # Update metadata labels
        self.label_scan_name.setText(f"Scan: {self.current_scan.name}")

        # Refresh the canvas
        self.canvas_plot.draw_idle()

    def start_phase_retrieval(self):
        """
        Start the phase retrieval in a separate thread and activate a timer.
        """

        # Disable the start button to prevent multiple clicks
        self.button_start_reconstruction.setEnabled(False)
        self.button_find_focus.setEnabled(False)
        self.button_stop_reconstruction.setEnabled(True)
        self.reconstruction_status_label.setText("Status: Running...")

        # Initialize the SinglePhaseRetrievalInitializer
        self.config.read(self.config_path)
        init_object_path = self.config["paths"]["params_dir"] + "/init_object.pkl"
        self.phase_retrieval_initializer = SinglePhaseRetrievalInitializer(init_object_path)

        # Create a new thread for phase retrieval
        self.phase_retrieval_thread = PhaseRetrievalThread(
            initializer=self.phase_retrieval_initializer,
            img_index=self.img_num,
            canvas=self.canvas,
        )

        # Check if already redirected
        if sys.stdout != self.original_stdout:
            self.original_stdout = sys.stdout
            self.original_stderr = sys.stderr

        # Redirect stdout and stderr to QTextEditLogger
        sys.stdout = QTextEditLogger(self.log_display)
        sys.stderr = QTextEditLogger(self.log_display)

        # Start the phase retrieval thread
        self.phase_retrieval_thread.start()

        # Connect the thread's finished signal to the handler
        self.phase_retrieval_thread.finished_signal.connect(self.on_reconstruction_finished)

        # Start the timer to periodically refresh the canvas
        self.refresh_timer.start()

    def update_canvas_during_reconstruction(self):
        """
        Periodically refresh the canvas during reconstruction.
        """
        self.canvas.draw_idle()

    def on_reconstruction_finished(self):
        """
        Called when the reconstruction thread signals completion.
        """
        self.refresh_timer.stop()

        # Restore stdout and stderr
        if hasattr(self, "original_stdout") and sys.stdout != self.original_stdout:
            sys.stdout = self.original_stdout
        if hasattr(self, "original_stderr") and sys.stderr != self.original_stderr:
            sys.stderr = self.original_stderr

        # Re-enable the start button and update status
        self.button_start_reconstruction.setEnabled(True)
        self.reconstruction_status_label.setText("Status: Completed")

        # Ensure the final result is drawn
        self.canvas.draw_idle()

    def open_advanced_settings(self):
        """
        Open the Advanced Settings window.
        """
        from gui.windows.popup_windows.advanced_json_editor import AdvancedJsonEditor

        config_path = os.path.join(
            self.current_scan.path_processed,
            "holopipe",
            "config",
            "holopipe_config.json",
        )
        print(config_path)
        print(self.current_scan)
        if not os.path.exists(config_path):
            QtWidgets.QMessageBox.warning(self, "No Config Found", "Please create the config first.")
            return

        self.advanced_json_window = AdvancedJsonEditor(config_path)
        self.advanced_json_window.show()

    def process_img_for_display(self, img_num):
        """
        Process and display the specified image in the Silx viewer.
        """
        img = self.current_scan.load_image(img_num - 1)
        img = np.flip(img, axis=0)
        img_rotated = np.rot90(img, k=-1)

        # Display the rotated image
        colormap = {"name": "gray", "autoscaleMode": "stddev3"}
        self.imv.setDefaultColormap(colormap)
        self.imv.addImage(img_rotated)

    def start_find_focus(self):
        """
        Start the focus finding process in a separate thread and update UI components.
        """
        # Disable the start button to prevent multiple clicks
        self.button_find_focus.setEnabled(False)
        self.button_start_reconstruction.setEnabled(False)
        self.button_stop_reconstruction.setEnabled(True)
        self.reconstruction_status_label.setText("Status: Running...")

        # Initialize the FindFocusInitializer
        self.config.read(self.config_path)
        init_object_path = self.config["paths"]["params_dir"] + "/init_object.pkl"
        self.find_focus_initializer = FindFocusInitializer(init_object_path)

        # Create a new thread for find focus
        self.find_focus_thread = FindFocusThread(
            initializer=self.find_focus_initializer,
            img_index=self.img_num,
            canvas=self.canvas,
        )

        # Check if already redirected
        if sys.stdout != self.original_stdout:
            self.original_stdout = sys.stdout
            self.original_stderr = sys.stderr

        # Redirect stdout and stderr to QTextEditLogger
        sys.stdout = QTextEditLogger(self.log_display)
        sys.stderr = QTextEditLogger(self.log_display)

        # Start the find focus thread
        self.find_focus_thread.start()

        # Connect the thread's finished signal to the handler
        self.find_focus_thread.finished_signal.connect(self.on_find_focus_finished)

        # Start the timer to periodically refresh the canvas
        self.refresh_timer.start()

    def stop_reconstruction(self):
        """
        Stop the current phase retrieval or focus finding process.
        """
        if self.phase_retrieval_thread and self.phase_retrieval_thread.isRunning():
            self.phase_retrieval_thread.terminate()
            self.on_reconstruction_finished()
        elif self.find_focus_thread and self.find_focus_thread.isRunning():
            self.find_focus_thread.terminate()
            self.on_find_focus_finished()

    def on_find_focus_finished(self):
        """
        Handle completion of the focus finding process.
        """
        self.refresh_timer.stop()
        self.button_start_reconstruction.setEnabled(True)
        self.button_find_focus.setEnabled(True)
        self.button_stop_reconstruction.setEnabled(False)
        self.reconstruction_status_label.setText("Status: Focus Finding Completed")

    def on_image_num_entered(self):
        """
        Update the image number entered in the text field.
        """
        try:
            new_img_num = int(self.image_num_input.text())
            if 1 <= new_img_num <= self.current_scan.num_img:
                self.img_num = new_img_num
                self.slider_num.setValue(self.img_num - 1)  # Sync with slider
                self.update_image_info()
                self.update_placeholder_image(self.current_scan.load_image(self.img_num - 1))
            else:
                QtWidgets.QMessageBox.warning(self, "Invalid Image Number", "Image number is out of range.")
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Input",
                "Please enter a valid integer for the image number.",
            )

    def clear_logging(self):
        """Clear the contents of the log display."""
        self.log_display.clear()

    def clear_iteration_canvas(self):
        """Clear the iteration canvas display."""
        self.fig.clear()
        self.canvas.draw_idle()

    def prev_tomo_image(self):
        """
        Navigate to the previous image in the selected view.
        """
        if self.slider_tomo.value() > 1:
            self.slider_tomo.setValue(self.slider_tomo.value() - 1)
            self.update_tomo_view()

    def next_tomo_image(self):
        """
        Navigate to the next image in the selected view.
        """
        if self.slider_tomo.value() < self.current_max_image:
            self.slider_tomo.setValue(self.slider_tomo.value() + 1)
            self.update_tomo_view()

    def update_tomo_view(self):
        """
        Update the Silx Viewer based on the current view and slider position.
        """
        current_index = self.slider_tomo.value()  # Get the current image index
        view_type = self.current_view  # Either 'slice', 'phase', or 'stack'

        if view_type == "slice":
            img_path = os.path.join(self.slice_folder, f"image_{current_index:04d}.tif")
        elif view_type == "phase":
            img_path = os.path.join(self.phase_folder, f"image_{current_index:04d}.tif")
        elif view_type == "stack":
            img_path = os.path.join(self.stack_folder, f"image_{current_index:04d}.tif")
        else:
            QtWidgets.QMessageBox.warning(self, "Error", "Invalid view type selected.")
            return

        # Load and display the image
        if os.path.exists(img_path):
            img = self.load_image(img_path)
            self.imv_tomo.clear()
            self.imv_tomo.addImage(img)
        else:
            QtWidgets.QMessageBox.warning(self, "Error", f"Image not found: {img_path}")

    def show_slice_view(self):
        """
        Set the viewer to show the reconstructed slices.
        """
        self.current_view = "slice"
        self.current_max_image = len(os.listdir(self.slice_folder))  # Adjust slider range
        self.slider_tomo.setRange(1, self.current_max_image)
        self.slider_tomo.setValue(1)  # Reset to the first image
        self.update_tomo_view()

    def show_phase_view(self):
        """
        Set the viewer to show the phase retrieved images.
        """
        self.current_view = "phase"
        self.current_max_image = len(os.listdir(self.phase_folder))  # Adjust slider range
        self.slider_tomo.setRange(1, self.current_max_image)
        self.slider_tomo.setValue(1)  # Reset to the first image
        self.update_tomo_view()

    def show_stack_view(self):
        """
        Set the viewer to show the reconstructed stack.
        """
        self.current_view = "stack"
        self.current_max_image = len(os.listdir(self.stack_folder))  # Adjust slider range
        self.slider_tomo.setRange(1, self.current_max_image)
        self.slider_tomo.setValue(1)  # Reset to the first image
        self.update_tomo_view()

    def update_tomo_image(self):
        """
        Update the displayed image in the viewer based on the slider value and active tab.
        """
        current_tab = self.viewer_tabs.currentWidget()
        current_slice = self.slider_tomo_slice.value() - 1  # Convert slider value to 0-based index

        if current_tab == self.slice_viewer_tab:
            if hasattr(self, "slice_stack") and self.slice_stack is not None:
                self.slice_image_viewer.clear()
                self.slice_image_viewer.addImage(self.slice_stack[current_slice])
        elif current_tab == self.phase_retrieved_tab:
            if hasattr(self, "phase_retrieved_stack") and self.phase_retrieved_stack is not None:
                self.phase_image_viewer.clear()
                self.phase_image_viewer.addImage(self.phase_retrieved_stack[current_slice])
        elif current_tab == self.stack_viewer_tab:
            if hasattr(self, "tomographic_stack") and self.tomographic_stack is not None:
                self.stack_image_viewer.clear()
                self.stack_image_viewer.addImage(self.tomographic_stack[current_slice])

    def update_tomo_slider(self):
        """
        Update the slider range and reset the current position based on the active tab.
        """
        current_tab = self.viewer_tabs.currentWidget()

        if current_tab == self.slice_viewer_tab:
            stack_size = self.slice_stack.shape[0] if hasattr(self, "slice_stack") else 1
        elif current_tab == self.phase_retrieved_tab:
            stack_size = self.phase_retrieved_stack.shape[0] if hasattr(self, "phase_retrieved_stack") else 1
        elif current_tab == self.stack_viewer_tab:
            stack_size = self.tomographic_stack.shape[0] if hasattr(self, "tomographic_stack") else 1
        else:
            stack_size = 1  # Fallback if no stack is loaded

        self.slider_tomo_slice.setRange(1, stack_size)
        self.slider_tomo_slice.setValue(1)  # Reset to the first slice
        self.update_tomo_image()

    def load_slice_stack(self, stack):
        """
        Load a new slice stack into the Slice Viewer tab.
        """
        self.slice_stack = stack
        self.update_tomo_slider()

    def load_phase_retrieved_stack(self, stack):
        """
        Load a new phase retrieved stack into the Phase Retrieved tab.
        """
        self.phase_retrieved_stack = stack
        self.update_tomo_slider()

    def load_tomographic_stack(self, stack):
        """
        Load a new tomographic stack into the Stack Viewer tab.
        """
        self.tomographic_stack = stack
        self.update_tomo_slider()

    # def update_img_index(self):
    #     """
    #     Dynamically update the img_index in the default.ini file.
    #     """
    #     try:
    #         update_ini_section(
    #             self.config_path,
    #             section='single_phase_retrieval',
    #             updates={
    #                 'img_index': self.img_num
    #             }
    #         )
    #     except Exception as e:
    #         QtWidgets.QMessageBox.warning(
    #             self, "Update Failed",
    #             f"An error occurred while updating default.ini: {e}"
    #         )

    def open_phase_viewer_popup(self):
        """
        Open a popup viewer for scrolling through phase retrieved images.
        """
        # Get the path from the .ini file
        self.config.read(self.config_path)
        phase_images_path = self.config["paths"]["reco_output"]

        if not os.path.exists(phase_images_path) or not os.listdir(phase_images_path):
            QtWidgets.QMessageBox.warning(
                self,
                "No Images Found",
                f"No phase retrieved images found in {phase_images_path}.",
            )
            return

        # Create the pop-up window
        self.phase_popup = QtWidgets.QWidget()
        self.phase_popup.setWindowTitle("Phase Retrieved Images Viewer")
        layout = QVBoxLayout(self.phase_popup)

        # Silx viewer for images
        self.phase_viewer = Plot2D()
        self.phase_viewer.setKeepDataAspectRatio(True)
        layout.addWidget(self.phase_viewer)

        # Slider for navigation
        slider_layout = QHBoxLayout()
        self.phase_slider = QSlider(Qt.Horizontal)
        self.phase_slider.setEnabled(True)
        slider_layout.addWidget(self.phase_slider)
        layout.addLayout(slider_layout)

        # Load images from the directory
        self.phase_images = sorted(
            [os.path.join(phase_images_path, f) for f in os.listdir(phase_images_path) if f.endswith(".tif")]
        )
        self.phase_slider.setRange(1, len(self.phase_images))
        self.phase_slider.setValue(1)
        self.phase_slider.valueChanged.connect(self.update_phase_image)

        # Show the first image
        self.update_phase_image()

        self.phase_popup.setLayout(layout)
        self.phase_popup.resize(800, 600)
        self.phase_popup.show()

    def update_phase_image(self):
        """
        Update the image displayed in the phase viewer based on the slider position.
        """
        if hasattr(self, "phase_images") and self.phase_images:
            image_index = self.phase_slider.value() - 1  # Slider is 1-based
            image_path = self.phase_images[image_index]

            if os.path.exists(image_path):
                img = self.load_image(image_path)
                self.phase_viewer.clear()
                self.phase_viewer.addImage(img)
            else:
                QtWidgets.QMessageBox.warning(self, "Error", f"Image not found: {image_path}.")

    def load_image(self, image_path):
        """
        Load an image from the specified path and return it as a NumPy array.
        """
        import tifffile

        try:
            return tifffile.imread(image_path)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to load image: {e}")
            return None
