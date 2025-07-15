from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QSplashScreen


class SplashScreen(QSplashScreen):
    """
    Splash screen that displays an image while the main window is loading.

    Attributes:
        splash_image (str): Path to the image to display on the splash screen.
    """

    def __init__(self, splash_image: str):
        """
        Initialize the splash screen with an image.

        Args:
            splash_image (str): The file path of the splash image.
        """
        # Load the image into a QPixmap object
        pixmap = QPixmap(splash_image)

        # Ensure the pixmap was loaded correctly (add a basic error check)
        if pixmap.isNull():
            raise FileNotFoundError(f"Could not load splash image at {splash_image}")

        # Resize the pixmap to a smaller size (e.g., 400x300)
        scaled_pixmap = pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Initialize the splash screen with the scaled image
        super().__init__(scaled_pixmap)

        # Set the splash screen to stay on top of other windows and have no window borders
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
