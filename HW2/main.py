from PyQt5 import QtWidgets, QtGui, QtCore
from Hw2_ui import Ui_Dialog
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import random
import time
import pickle

import csv
import sys
import numpy as np
import cv2 as cv
import glob

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        # the init of mainwindows
        super(MainWindow, self).__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        
        #the init of Q1
        self.ui.pushButton.setText('1.1 Background Subtraction')
        self.ui.pushButton.clicked.connect(self.Problem1_1)

        #the init of Q2
        self.frame = None
        self.points = None
        self.ui.pushButton_2.setText('2.1 Preprocessing')
        self.ui.pushButton_2.clicked.connect(self.Problem2_1)
        self.ui.pushButton_3.setText('2.2 Video tracking')
        self.ui.pushButton_3.clicked.connect(self.Problem2_2)

        #the init of Q3
        self.ui.pushButton_4.setText('3.1 Perspective Transform')
        self.ui.pushButton_4.clicked.connect(self.Problem3_1)

        #the init of Q4
        self.reconstruction_error = []
        self.ui.pushButton_5.setText('4.1 Image Reconstruction')
        self.ui.pushButton_5.clicked.connect(self.Problem4_1)
        self.ui.pushButton_6.setText('4.2 Compute the Reconstruction Error')
        self.ui.pushButton_6.clicked.connect(self.Problem4_2)
        
        #the init of Q5
        self.ui.pushButton_7.setText('5.1 Resnet50_show_accuracy')
        self.ui.pushButton_7.clicked.connect(self.Problem5_1)
        self.ui.pushButton_8.setText('5.2 Resnet50_show_tensorboard_training')
        self.ui.pushButton_8.clicked.connect(self.Problem5_2)
        self.ui.pushButton_9.setText('5.3 Resnet50_random_select_testing_image')
        self.ui.pushButton_9.clicked.connect(self.Problem5_3)
        self.ui.pushButton_10.setText('5.4 Resnet50_show_tensorboard_training')
        self.ui.pushButton_10.clicked.connect(self.Problem5_4)
        
    
    def Problem1_1(self):
        vedio=cv.VideoCapture("Q1_Image/bgSub.mp4")

        fps=vedio.get(cv.CAP_PROP_FPS)

        frame_data=[]
        while vedio.isOpened():
            ret, frame=vedio.read()
            if not ret:
                break
            frame_data.append(frame)

        train_frame=frame_data[:50]
        test_frame=frame_data[50:]

        model=Subtractor(train_frame)

        for frame in test_frame:
            subframe=model.match(frame)

            subframe_stack=np.stack((subframe,) * 3, axis=-1)
            resultframe=cv.hconcat([frame,subframe_stack])
            cv.imshow("Demo",resultframe)

            if cv.waitKey(50) & 0xFF == ord("q"):
                break
        
    
    
    def point(self):
        vedio=cv.VideoCapture("Q2_Image/opticalFlow.mp4")

        first=None

        while  vedio.isOpened() and first is None:
            ret, frame=vedio.read()
            first=frame

        vedio.release()

        parameter=cv.SimpleBlobDetector_Params()

        parameter.filterByCircularity=True
        parameter.minCircularity=0.81

        sensor=cv.SimpleBlobDetector_create(parameter)
        keypoints=sensor.detect(frame)

        pts=[]
        for i in keypoints:
            x, y=i.pt
            x=round(x)
            y=round(y)
            if x < 500:
                pts.append(i.pt)
                frame=cv.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), 1)
        self.points=np.asarray(pts).reshape((len(pts), 1, 2))
        self.frame=frame

    def Problem2_1(self):
        
        self.point()
        cv.imshow("Keypoints", self.frame)
        cv.waitKey(0)


    def Problem2_2(self):

        if self.frame is None:
            self.get_points()

        vedio=cv.VideoCapture("Q2_Image/opticalFlow.mp4")
        ret, frame=vedio.read()
        lk_params=dict(winSize=(21, 21),maxLevel=2,
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),)

        color=np.random.randint(0, 255, 3)

        old_frame=self.frame
        old_gray=cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

        p0=self.points.astype(np.float32)

        mask=np.zeros_like(old_frame)
        while 1:
            try:
                ret, frame=vedio.read()
                frame_gray=cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            except:
                break

            p1, st, err=cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            good_new=p1[st == 1]
            good_old=p0[st == 1]
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                x, y=new.ravel()
                a, b=old.ravel()
                mask=cv.line(mask, (x, y), (a, b), color.tolist(), 1)
                x=int(round(x))
                y=int(round(y))
                frame=cv.rectangle(frame, (x-5, y-5), (x+5, y+5), (100, 100, 255), 1)
                frame=cv.line(frame, (x-5, y), (x+5, y), (100, 100, 255), 1)
                frame=cv.line(frame, (x, y-5), (x, y+5), (100, 100, 255), 1)

            img=cv.add(frame, mask)
            cv.imshow("frame", img)
            k=cv.waitKey(30) & 0xFF
            if k==27:
                break
            old_gray=frame_gray.copy()
            p0=good_new.reshape(-1, 1, 2)

        vedio.release()


    def Problem3_1(self):

        vedio=cv.VideoCapture("Q3_Image/test4perspective.mp4")
        source=cv.imread("Q3_Image/rl.jpg")
        height, width, _=source.shape

        d=cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)
        parameter=cv.aruco.DetectorParameters_create()
        id_sequence=[25, 33, 30, 23]
        edges=[1, 2, 0, 0]

        while vedio.isOpened():
            ret, frame = vedio.read()
            if not ret:
                break
            try:
                (
                    marker_corners,
                    marker_ids,
                    rejected_candidates,
                ) = cv.aruco.detectMarkers(frame, d, parameters=parameter)

                edge_points=[]
                for i, marker_id in enumerate(id_sequence):
                    index=np.squeeze(np.where(marker_ids == marker_id))[0]
                    edge_points.append(np.squeeze(marker_corners[index])[edges[i]])

                distance=np.linalg.norm(edge_points[0] - edge_points[1])
                offset=round(distance * 0.02)

                operators=[[-1, -1], [1, -1], [1, 1], [-1, 1]]
                for i, operator in enumerate(operators):
                    edge_points[i][0] += operator[0] * offset
                    edge_points[i][1] += operator[1] * offset

                source_points=np.array([[0, 0], [width, 0], [width, height], [0, height]])

                M, mask=cv.findHomography(source_points, np.array(edge_points), cv.RANSAC)

                w, h, _=frame.shape
                mask_img=cv.warpPerspective(source, M, (h, w))
                ret, mask=cv.threshold(cv.cvtColor(mask_img, cv.COLOR_BGR2GRAY),
                    0,255,cv.THRESH_BINARY_INV,)
                new_frame=cv.bitwise_and(frame, frame, mask=mask)
                new_frame=cv.bitwise_or(new_frame, mask_img)

                cv.imshow("new_frame", new_frame)
            except:
                pass
            if cv.waitKey(50) & 0xFF == ord("q"):
                break


    def Problem4_1(self):
        images=[]

        badges=[]
        for i in range(1,35):
            image=cv.imread(f"Q4_Image/{i}.jpg")
            badges.append(image.flatten())

        images_reduce=[]
        for i in range(1,35):
            image=cv.imread(f"Q4_Image/{i}.jpg")

            print(f"processing image {i}")

            image_size=(100, 100, 3)

            estimator=PCA(n_components=int(0.8 * 34))
            estimator.fit(badges)

            components=estimator.transform(badges)
            projected=estimator.inverse_transform(components)

            reduced=projected[i-1]
            reduced=(reduced - reduced.min()) * 255 / (reduced.max() - reduced.min())
            reduced=reduced.astype(np.uint8)

            error=np.sum((badges[i-1] - reduced) ** 2)
            self.reconstruction_error.append(error)

            image_reduce=reduced.reshape(image_size)
            images_reduce.append(image_reduce)

        # for i, image in enumerate(images_reduce):
        for i in range(len(images_reduce)):
            plt.subplot(4, 17, int(i/17)*34+i%17+1)
            plt.imshow(cv.cvtColor(badges[i].reshape(image_size), cv.COLOR_BGR2RGB))
            plt.xticks(())
            plt.yticks(())
            plt.subplot(4, 17, int(i/17)*34+i%17+17+1)
            plt.imshow(cv.cvtColor(images_reduce[i], cv.COLOR_BGR2RGB))
            plt.xticks(())
            plt.yticks(())
        plt.show()

        cv.waitKey(0)

    def Problem4_2(self):
        print(self.reconstruction_error)
    
    def Problem5_1(self):


        with open('./resnet50/output/resnet50_train_history_log.csv', 'r') as history_file:
            reader = csv.DictReader(history_file, delimiter=',')
            title = reader.fieldnames
            for titleItem in title:
                print(titleItem, end="  ")
            print("")

            for row in reader:
                print("%s      %s      %s  %s           %s" % (
                        row["epoch"], row["accuracy"][0:4],
                        row["loss"][0:4], row["val_accuracy"][0:4],
                        row["val_loss"][0:4]))

    def Problem5_2(self):
        tb_acc_loss = plt.imread("./resnet50/output/output.png")

        plt.title("Tensorboard Training Process")
        plt.imshow(tb_acc_loss)
        plt.show()

    def Problem5_3(self):
        idx_all = random.randint(0, 24999)

        plt.figure(figsize=(1,1))
        if idx_all >= 12500:
            new_idx = idx_all // 2 # Integer division

            # Select picture(s) from Dog set
            img = plt.imread("./resnet50/dataset_ASIRRA/PetImages/Dog/%d.jpg" % new_idx)
            plt.title("Dog(1)")
        else:
            new_idx = idx_all

            # Select picture(s) from Cat set
            img = plt.imread("./resnet50/dataset_ASIRRA/PetImages/Cat/%d.jpg" % new_idx)
            plt.title("Cat(0)")

        # figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()

        plt.imshow(img)
        plt.show()

    
    def Problem5_4(self):
        pass


class Subtractor:
    def __init__(self, train_frame):
        self.train(train_frame)

    def train(self, train_frame):
        frame=np.array([cv.cvtColor(frame, cv.COLOR_RGB2GRAY) for frame in train_frame])

        self.pixels=np.moveaxis(frame, 0, -1)
        height, width, n=self.pixels.shape
        self.mean=np.zeros((height, width))
        self.std=np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                pixels=self.pixels[i][j]
                mean=np.mean(pixels)
                std=np.std(pixels)
                std=std if std > 5 else 5
                self.mean[i][j]=mean
                self.std[i][j]=std

    def match(self, frame):
        gray=cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

        new_frame=gray - self.mean
        new_frame=(new_frame > 5 * self.std).astype(np.uint8) * 255

        return new_frame

        


if __name__ == '__main__':
     app = QtWidgets.QApplication([])
     window = MainWindow()
     window.show()
     sys.exit(app.exec_())