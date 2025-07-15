from PyQt5.QtCore import QThread, pyqtSignal

class FindFocusThread(QThread):
    """
    A thread for running a find focus process asynchronously. Emits a signal
    when the process is complete, allowing for non-blocking execution within the GUI.

    Attributes:
        finished_signal (pyqtSignal): Signal emitted when the phase retrieval is finished.
        initializer: Object containing the find focus function.
        img_index (int): Index of the image to be processed.
        canvas: GUI canvas or widget where the processed image will be displayed.
    """

    finished_signal = pyqtSignal()  # Signal to indicate completion

    def __init__(self, initializer, img_index, canvas):
        """
        Initializes the FindFocusThread with necessary parameters for find focus.

        Args:
            initializer: Object or instance with the `single_reco_gui` method for phase retrieval.
            img_index (int): The index of the image to process.
            canvas: The GUI canvas or widget where the image will be displayed.
        """
        super().__init__()
        self.initializer = initializer
        self.img_index = img_index
        self.canvas = canvas

    def run(self):
        """
        Executes the find focus process by calling the `find_focus_gui` method
        from the initializer. Emits `finished_signal` upon completion.
        """
        self.initializer.run_focus_gui(self.canvas, img_index=self.img_index)
        self.finished_signal.emit()  # Signal that the process is complete
