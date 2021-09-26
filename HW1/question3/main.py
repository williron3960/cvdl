from PyQt5 import QtWidgets, QtGui, QtCore
from UI_question3 import Ui_Dialog
import sys
import cv2 as cv
import matplotlib.pyplot as plt


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)


        self.ui.pushButton.setText('3. Disparity Map')
        self.ui.pushButton.clicked.connect(self.buttonClicked)
        self.imgL = cv.imread('./Q3_Image/imL.png', 0)
        self.imgR = cv.imread('./Q3_Image/imR.png', 0)

    def buttonClicked(self):
        stereo = cv.StereoSGBM_create(
            minDisparity = 0,
            numDisparities = 16,
            blockSize = 9,
            P1 = 64*3*9,
            P2 = 256*3*9,
            uniquenessRatio = 10,
            speckleWindowSize = 100,
            speckleRange = 32
        )

        disparity = stereo.compute(self.imgL, self.imgR)
        plt.imshow(disparity, 'gray')
        plt.show()

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())