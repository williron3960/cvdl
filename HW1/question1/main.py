from PyQt5 import QtWidgets, QtGui, QtCore
from UI_interface import Ui_Dialog
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
        self.files = [f'./Q1_Image/{i}.bmp' for i in range(1,16,1)]
        self.inst_mat = None
        self.tvecs = []
        self.dist = None

        self.ui.pushButton.setText('1.1 Find Corners')
        self.ui.pushButton.clicked.connect(self.buttonClicked)
        self.ui.pushButton_2.setText('1.2 Find Intrinsic')
        self.ui.pushButton_2.clicked.connect(self.buttonClicked_2)
        self.ui.pushButton_3.setText('1.4 Find Distortion')
        self.ui.pushButton_3.clicked.connect(self.buttonClicked_3)
        self.ui.comboBox.addItem(" ")
        self.ui.label.setText('Select Image')
        self.ui.comboBox.addItems(self.files)
        self.ui.comboBox.activated[str].connect(self.Now_image)
        self.ui.pushButton_4.setText('1.3 Extrinsic Matrix')
        self.ui.pushButton_4.clicked.connect(self.Extrinsic_matrix)

    def Now_image(self, file):
        self.Now_filename = file
        print(f"Selected {file}")
        self.index = self.files.index(file)


    def buttonClicked(self):
        filename = './Q1_Image/*.bmp'
    #    img = cv2.imread(filename)
        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((self.img_w * self.img_h, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.img_w, 0:self.img_h].T.reshape(-1, 2)
        # Arrays to store object points and image points from all the images.
        images = glob.glob(filename)
        for fname in images:
            img = cv.imread(fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (self.img_w, self.img_h), None)
            if ret == True:
                self.objp.append(objp)
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                self.imgp.append(corners)

                cv.drawChessboardCorners(img, (self.img_w,self.img_h), corners2, ret)
                cv.imshow(fname, img)
                cv.waitKey(1000)
        cv.destroyAllWindows()
    
    def Extrinsic_matrix(self):
        print('Current:', self.Now_filename)
        if self.index >= 0:
            print('Extrinsic matrix : ')
            print(np.concatenate((self.rvecs[self.index], self.tvecs[self.index]), axis=1))

    def buttonClicked_3(self):
        print('Distortion Matrix:')
        print(self.dist)


    def buttonClicked_2(self):
        filename = './Q1_Image/*.bmp'
        #    img = cv2.imread(filename)
        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((self.img_w * self.img_h, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.img_w, 0:self.img_h].T.reshape(-1, 2)
        images = glob.glob(filename)
        for fname in images:
            img = cv.imread(fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
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
            self.rvecs = [cv.Rodrigues(rvecs[i])[0] for i in range(15)]
            self.tvecs = tvecs
            print('Intrinsic Matrix:')
            print(mtx)





if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
