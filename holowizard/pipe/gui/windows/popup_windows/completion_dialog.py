from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel


class CompletionDialog(QDialog):
    """
    A simple dialog window that displays a message for a set period of time
    and then automatically closes.

    Attributes:
        message (str): The message to display in the dialog.
        window_title (str): The title to display on the dialog window.
        parent (QWidget, optional): The parent widget, if any. Defaults to None.
    """

    def __init__(self, message, window_title, parent=None):
        """
        Initializes the CompletionDialog with a message and title.

        Args:
            message (str): The message to display in the dialog.
            window_title (str): The title to display on the dialog window.
            parent (QWidget, optional): The parent widget, if any. Defaults to None.
        """
        super().__init__(parent)
        self.setWindowTitle(window_title)
        self.resize(300, 100)

        # Set up the layout and add the message label
        layout = QVBoxLayout(self)
        label = QLabel(message)
        layout.addWidget(label)

        # Close the dialog after 3 seconds
        QTimer.singleShot(3000, self)
