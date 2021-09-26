from PyQt5 import QtWidgets, QtGui, QtCore
from UI_question2 import Ui_Dialog
import sys
import numpy as np
import cv2 as cv
import glob

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.rvecs =[]
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.img_s = 0
        self.img_w = 11
        self.img_h = 8
        self.objp = []
        self.index=-1
        self.imgp = []
        self.instrinc_matrix = None

        self.files = [f'./Q2_Image/{i}.bmp' for i in range(1,6,1)]
        self.inst_mat = None
        self.tvecs = []
        self.dist = None

        self.ui.pushButton.setText('2. Augmented Reality')
        self.ui.pushButton.clicked.connect(self.buttonClicked)

    def buttonClicked(self):
        filename = './Q2_Image/*.bmp'
        #    img = cv2.imread(filename)
        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((self.img_w * self.img_h, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.img_w, 0:self.img_h].T.reshape(-1, 2)
        # images = glob.glob(filename)
        for fname in self.files:
            img = cv.imread(fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            height, width, channels = img.shape
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (self.img_w, self.img_h), None)
            if ret == True:
                self.objp.append(objp)
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                self.imgp.append(corners)
        self.img_s = gray.shape[::-1]
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(self.objp, self.imgp, self.img_s, None, None)
        if ret:
            self.instrinc_matrix = mtx
            self.dist = dist
            self.rvecs = [cv.Rodrigues(rvecs[i])[0] for i in range(5)]
            self.tvecs = tvecs
            print('Intrinsic Matrix:')
            print(mtx)
            print('distortion Matrix:')
            print(dist)
        for file in self.files:
            self.index = self.files.index(file)
            print('Current:'+file)
            if self.index >= 0:
                print('Extrinsic matrix : ')
                print(np.concatenate((self.rvecs[self.index], self.tvecs[self.index]), axis=1))






if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())