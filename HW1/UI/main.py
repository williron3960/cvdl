from ui import Ui_Dialog
from PyQt5 import QtWidgets, QtGui, QtCore
import sys
import numpy as np
import cv2 as cv
import glob
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        self.ui.pushButton.setText('2. Augmented Reality')
        self.ui.pushButton.clicked.connect(self.buttonClicked)

    def buttonClicked(self):
        print('hello world')

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())