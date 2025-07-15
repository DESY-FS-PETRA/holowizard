from PyQt5 import QtWidgets


def create_divider():
    """
    Helper function to create a horizontal line divider
    """
    line = QtWidgets.QFrame()
    line.setFrameShape(QtWidgets.QFrame.HLine)
    line.setFrameShadow(QtWidgets.QFrame.Sunken)
    return line
