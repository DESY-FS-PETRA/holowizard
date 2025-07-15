from PyQt5.QtCore import QThread, pyqtSignal

class PhaseRetrievalThread(QThread):
    """
    A thread for running a phase retrieval process asynchronously. Emits a signal
    when the process is complete, allowing for non-blocking execution within the GUI.

    Attributes:
        finished_signal (pyqtSignal): Signal emitted when the phase retrieval is finished.
        initializer: Object containing the phase retrieval function.
        img_index (int): Index of the image to be processed.
        canvas: GUI canvas or widget where the processed image will be displayed.
    """

    finished_signal = pyqtSignal()  # Signal to indicate completion

    def __init__(self, initializer, img_index, canvas):
        """
        Initializes the PhaseRetrievalThread with necessary parameters for phase retrieval.

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
        Executes the phase retrieval process by calling the `single_reco_gui` method
        from the initializer. Emits `finished_signal` upon completion.
        """
        self.initializer.single_reco_gui(self.canvas, img_index=self.img_index)
        self.finished_signal.emit()  # Signal that the process is complete
