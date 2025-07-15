from PyQt5.QtCore import QMetaObject, Qt, Q_ARG

class QTextEditLogger:
    """
    A logger that directs text output to a QTextEdit widget, allowing messages
    to be appended to the widget asynchronously. This class is useful for
    redirecting standard output or custom logging messages to a GUI text editor.

    Attributes:
        widget (QTextEdit): The QTextEdit widget where messages will be displayed.
    """

    def __init__(self, text_edit_widget):
        """
        Initializes the QTextEditLogger with the specified QTextEdit widget.

        Args:
            text_edit_widget (QTextEdit): The QTextEdit widget to display log messages.
        """
        self.widget = text_edit_widget

    def write(self, message):
        """
        Appends a message to the QTextEdit widget, ensuring it is executed
        in the main GUI thread.

        Args:
            message (str): The message to be displayed. Empty messages are skipped.
        """
        if message.strip():  # Skip empty lines
            QMetaObject.invokeMethod(
                self.widget,
                "append",
                Qt.QueuedConnection,
                Q_ARG(str, message)
            )

    def flush(self):
        """
        Flush method to comply with file-like object requirements.

        This is a placeholder method; it does not perform any action in this context
        but is required to maintain compatibility with file-like objects.
        """
        pass
