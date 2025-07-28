import sys

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
from gui.windows.splash_screen import SplashScreen
from gui.windows.main_window import MainWindow


def main():
    """
    Main function to launch the application.
    Starts with the splash screen and then transitions to the main window.
    """
    app = QApplication(sys.argv)

    # Path to the splash screen image (adjust this if needed)
    splash_image_path = "../gui/resources/splash.png"

    # Create and display the splash screen
    splash = SplashScreen(splash_image_path)
    splash.show()

    # Create the main window, but don't show it yet
    main_window = MainWindow()

    # Function to transition from the splash screen to the main window
    def show_main_window():
        splash.close()
        main_window.show()

    QTimer.singleShot(3000, show_main_window)

    # Start the application event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
