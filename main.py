import cv2
import sys
import numpy
import abdomen_mask
from PyQt5 import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from math import *
from PyQt5 import QtCore
from random import *
from statistics import *
import numpy as np

weights_path = 'abs_model.h5'

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.mask_obj = abdomen_mask.Abd_Mask()
        self.filename = 'abs.jpg'
        self.height = 400
        self.width = 400
        self.initUI()
        
    def get_echo_lvl(self):
        
        path = 'Images/' + str(randint(0,14)) + '.png'
        frame = cv2.imread(path)
        old_frame = frame
        pix = QPixmap(path)
        self.im_label.setPixmap(pix)
        # Get abdomen section
        x,y,img = self.mask_obj.get_abdomen_mask(self.filename,frame,weights_path,self.height,self.width)
        ### Preprocessing
        cv2.imshow('before', img)
        cv2.imwrite('newest.jpg', img)
        img = cv2.imread('newest.jpg',cv2.IMREAD_GRAYSCALE)
        abdomen = cv2.medianBlur(img, 13) # Remove salt & pepper
        lap = cv2.Laplacian(abdomen, cv2.CV_8UC1, ksize=5) # Edge detection
        dil = cv2.dilate(lap, (3, 3)) # Enhance circles
        circles = cv2.HoughCircles(dil, cv2.HOUGH_GRADIENT, 16, 55, param2=450, minRadius=3, maxRadius=40) # Get circles

        if circles is not None:
            for i in circles[0, :]:
                # delete stomach and other hypo-echogenic regions
                print(i[0])
                cx = int(i[0])
                cy = int(i[1])
                dil[cx-10:cx+10, cy-10:cy+10] = 0
        else:
            print("No circles detected")
        # Calculating discrete grading of bowel hyperechogenicity whilst ignoring bones or empty spaces
        val = dil >0
        val = val < 1
        cv2.imshow('after',dil)
        discrete = [0, 1/6, 2/6, 3/6, 4/6, 5/6, 6/6]
        res = (np.sum(np.mean(val)))
        for i in range(len(discrete)-1):
            if res>=discrete[i] and res < discrete[i+1]:
                num = i
        print(res)
        self.count_label.setText("Hyperechogenicity grading: " + str(num))
        self.count_label.setFont(QFont("Times", 12, QFont.Bold))
        
    def initUI(self):
        self.setGeometry(5, 5, 1280, 720)
        self.resize(1280, 720)
        self.im_label = QLabel(self)
        self.im_label.setGeometry(QtCore.QRect(350, 200, 400, 400))
        self.load_btn = QPushButton(self)
        self.load_btn.setText("Load Image")
        self.load_btn.setGeometry(QtCore.QRect(50, 300, 200, 100))
        self.load_btn.clicked.connect(self.get_echo_lvl)
        self.count_label = QLabel(self)
        self.count_label.setGeometry(QtCore.QRect(800, 150, 400, 400))
        
        self.show()
        
if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    window = App()
    sys.exit(app.exec_())
        
