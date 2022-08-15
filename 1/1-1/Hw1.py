import sys
import cv2
import os
import numpy as np
import glob
from numpy.core.fromnumeric import size
from scipy import signal
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox
from hw1UI import Ui_MainWindow
import pickle
from matplotlib import pyplot as plt

class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.BindComponents()
        self.objpoints = None # 3d point in real world space
        self.imgpoints = None # 2d points in image plane
        self.intrinsicMatrix = None
        self.extrinsicMatrix = None
        self.distortionMatrix = None
        self.Q1_Image = self.LoadImage('./Dataset_CvDl_Hw1/Q1_Image/')
        self.Q2_Image = self.LoadImage('./Dataset_CvDl_Hw1/Q2_Image/')
        self.Q3_Image = self.LoadImage('./Dataset_CvDl_Hw1/Q3_Image/')

        self.img1 = cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/1.bmp')
        self.img2 = cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/2.bmp')
        self.img3 = cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/3.bmp')
        self.img4 = cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/4.bmp')
        self.img5 = cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/5.bmp')
        self.img6 = cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/6.bmp')
        self.img7 = cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/7.bmp')
        self.img8 = cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/8.bmp')
        self.img9 = cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/9.bmp')
        self.img10 = cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/10.bmp')
        self.img11 = cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/11.bmp')
        self.img12 = cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/12.bmp')
        self.img13 = cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/13.bmp')
        self.img14 = cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/14.bmp')
        self.img15 = cv2.imread('./Dataset_CvDl_Hw1/Q1_Image/15.bmp')
        self.img16 = cv2.imread('./Dataset_CvDl_Hw1/Q2_Image/1.bmp')
        self.img17 = cv2.imread('./Dataset_CvDl_Hw1/Q2_Image/2.bmp')
        self.img18 = cv2.imread('./Dataset_CvDl_Hw1/Q2_Image/3.bmp')
        self.img19 = cv2.imread('./Dataset_CvDl_Hw1/Q2_Image/4.bmp')
        self.img20 = cv2.imread('./Dataset_CvDl_Hw1/Q2_Image/5.bmp')
        self.img21 = cv2.imread('./Dataset_CvDl_Hw1/Q3_Image/imL.png')
        self.img22 = cv2.imread('./Dataset_CvDl_Hw1/Q3_Image/imR.png')
        self.img23 = cv2.imread('./Dataset_CvDl_Hw1/Q4_Image/Shark1.jpg')
        self.img24 = cv2.imread('./Dataset_CvDl_Hw1/Q4_Image/Shark2.jpg')
    
    def LoadImage(self, path):
        images = []
        files = os.listdir(path)
        files.sort()
        for f in files:
            image = cv2.imread(path + f)
            images.append(image)
        return images

    def BindComponents(self):
        self.btn1_1.clicked.connect(self.on_btn1_1_click)
        self.btn1_2.clicked.connect(self.on_btn1_2_click)
        self.btn1_3.clicked.connect(self.on_btn1_3_click)
        self.btn1_4.clicked.connect(self.on_btn1_4_click)
        self.btn1_5.clicked.connect(self.on_btn1_5_click)
        self.btn2_1.clicked.connect(self.on_btn2_1_click)
        self.btn2_2.clicked.connect(self.on_btn2_2_click)
        self.btn3_1.clicked.connect(self.on_btn3_1_click)
        self.btn4_1.clicked.connect(self.on_btn4_1_click)
        self.btn4_2.clicked.connect(self.on_btn4_2_click)
        self.btn4_3.clicked.connect(self.on_btn4_3_click)

    
    def on_btn1_1_click(self):
        #Enter the number of inside corners in x
        nx = 11
        #Enter the number of inside corners in y
        ny = 8
        # Make a list of calibration images
        chess_images = glob.glob('./Dataset_CvDl_Hw1/Q1_Image/*.bmp')
        # Select any index to grab an image from the list
        for i in range(len(chess_images)):
            # Read in the image
            chess_board_image = cv2.imread(chess_images[i])
            # Convert to grayscale
            gray = cv2.cvtColor(chess_board_image, cv2.COLOR_RGB2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(chess_board_image, (nx, ny), None)
            # If found, draw corners
            if ret == True:
                # Draw and display the corners
                cv2.drawChessboardCorners(chess_board_image, (nx, ny), corners, ret)
                result_name = 'board'+str(i+1)+'.jpg'
                chess_board_image = cv2.resize(chess_board_image, (1024, 1024))
                cv2.imshow(result_name, chess_board_image)
                cv2.waitKey(500)
                cv2.destroyAllWindows()
    
    def FindCorner(self, images, p = False):
        objpoints = []
        imgpoints = []

        objp = np.zeros((11 * 8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)

        i = 1
        for image in images:
            grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(grayImage, (11, 8), None)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(grayImage, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                imgpoints.append(corners2)
                if p:
                    img = image.copy()
                    cv2.drawChessboardCorners(img, (11, 8), corners2, ret)
                    img = cv2.resize(img, (1024, 1024))
                    cv2.imshow('Result ' + str(i), img)
                    cv2.waitKey(500)
                    cv2.destroyAllWindows()
            i += 1
        return objpoints, imgpoints

    def FindParameters(self, objpoints, imgpoints, img_size):      
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        # mtx = intrinsic matrix 
        # dist = distortion matrix
        # rvecs = rotation matrix
        # tvecs = translation matrix
        # rvecs + tvecs = extrinsic matrix
        intrinsicMatrix = mtx
        distortionMatrix = dist
        extrinsicMatrix = []
        for i in range(len(rvecs)):
            rvec, temp = cv2.Rodrigues(rvecs[i])
            extrinsicMatrix.append(np.concatenate((rvec, tvecs[i]), axis=1))
        return intrinsicMatrix, extrinsicMatrix, distortionMatrix, rvecs, tvecs

    def CheckParameters(self):
        if self.intrinsicMatrix is None or self.extrinsicMatrix is None or self.distortionMatrix is None or self.rvecs is None or self.tvecs:
            if self.objpoints is None or self.imgpoints is None:
                self.objpoints, self.imgpoints = self.FindCorner(self.Q1_Image)
            self.intrinsicMatrix, self.extrinsicMatrix, self.distortionMatrix, self.rvecs, self.tvecs = self.FindParameters(self.objpoints, self.imgpoints, (self.Q2_Image[0].shape[1], self.Q2_Image[0].shape[0]))

    def on_btn1_2_click(self):
        self.CheckParameters()
        print("Intrinsic:")
        print(self.intrinsicMatrix)

    def on_btn1_3_click(self):
        if self.Input1_1.text().isdigit() == False:
            QMessageBox.about(self, "檢查輸入", "請輸入1-15")
            return
        elif int(self.Input1_1.text())  >15 or int(self.Input1_1.text())  < 1:
            QMessageBox.about(self, "檢查輸入", "請輸入1-15")
            return
        self.CheckParameters()
        n = int(self.Input1_1.text())
        print(self.extrinsicMatrix[n])
    
    def on_btn1_4_click(self):
        self.CheckParameters()
        print("Distortion")
        print(self.distortionMatrix)

    def on_btn1_5_click(self):
        self.CheckParameters()
        mtx = self.intrinsicMatrix
        dist = self.distortionMatrix
        chess_images = glob.glob('./Dataset_CvDl_Hw1/Q1_Image/*.bmp')
        for i in range(len(chess_images)):
            # Read in the image
            chess_board_image = cv2.imread(chess_images[i])
            h,  w = chess_board_image.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
            img_size = (chess_board_image.shape[1],chess_board_image.shape[0])
            dst = cv2.undistort(chess_board_image, mtx, dist, None, newcameramtx)
            chess_board_image = cv2.resize(chess_board_image, (1024, 1024))
            dst = cv2.resize(dst, (1024, 1024))
            cv2.imshow('Original Image'+str(i+1), chess_board_image)
            cv2.imshow('Undistortion Image'+str(i+1), dst)
            cv2.waitKey(500)
            cv2.destroyAllWindows()

    def draw(self, img, corners, imgpts, number):
        corners = corners.astype(int)
        imgpts = imgpts.astype(int)
        corner = tuple(corners[number].ravel())
        for i in range(0,len(imgpts),2):
            #corner = tuple(corners[[i]].ravel())
            img = cv2.line(img,tuple(imgpts[i].ravel()),tuple(imgpts[i+1].ravel()), (255,0,0), 5)
            #img = cv2.line(img,corner,tuple(imgpts[i+1].ravel()), (255,0,0), 5)
        return img

    def on_btn2_1_click(self):
        fs = cv2.FileStorage('./Dataset_CvDl_Hw1/Q2_Image/Q2_Lib/alphabet_lib_onboard.txt', cv2.FILE_STORAGE_READ)
        if len(self.Input2_1.text()) <= 0 or len(self.Input2_1.text()) > 6:
            QMessageBox.about(self, "檢查輸入", "請輸入1-6個字")
            return
        
        chess_images = glob.glob('./Dataset_CvDl_Hw1/Q2_Image/*.bmp')
        word_arr = []
        for j in self.Input2_1.text():
            ch = fs.getNode(j).mat()
            word_arr.append(ch)
            #print(ch)

        for i in range(1,6):
            img = cv2.imread(chess_images[i-1])
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(img, (11, 8), None)
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            word_point = [62, 59, 56, 29, 26, 23]
            word_point = np.array(word_point)
            self.objpoints, self.imgpoints = self.FindCorner(self.Q1_Image)
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, (self.Q2_Image[0].shape[1], self.Q2_Image[0].shape[0]), None, None)
            self.CheckParameters()
            objp = np.zeros((11 * 8, 3), np.float32)
            objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
            ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
            rvecs = np.float32(rvecs)
            tvecs = np.float32(tvecs)
            x = [[7,5,0],[4,5,0],[1,5,0],[7,2,0],[4,2,0],[1,2,0]]

            for j in range(len(word_arr)):
                line = np.float32(word_arr[j] + x[j]).reshape(-1, 3)
                imgpts, jac = cv2.projectPoints(line, rvecs, tvecs, mtx, dist)
                img = self.draw(img, corners2, imgpts, word_point[j])
            img = cv2.resize(img,  (1024, 1024))
            cv2.imshow("windows_name", img)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()

    
    def on_btn2_2_click(self):
        fs = cv2.FileStorage('./Dataset_CvDl_Hw1/Q2_Image/Q2_Lib/alphabet_lib_vertical.txt', cv2.FILE_STORAGE_READ)
        if len(self.Input2_1.text()) <= 0 or len(self.Input2_1.text()) > 6:
            QMessageBox.about(self, "檢查輸入", "請輸入1-6個字")
            return
        
        chess_images = glob.glob('./Dataset_CvDl_Hw1/Q2_Image/*.bmp')
        word_arr = []
        for j in self.Input2_1.text():
            ch = fs.getNode(j).mat()
            word_arr.append(ch)
            #print(ch)

        for i in range(0,5):
            img = cv2.imread(chess_images[i])
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(img, (11, 8), None)
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            word_point = [62, 59, 56, 29, 26, 23]
            word_point = np.array(word_point)
            self.objpoints, self.imgpoints = self.FindCorner(self.Q1_Image)
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, (self.Q2_Image[0].shape[1], self.Q2_Image[0].shape[0]), None, None)
            self.CheckParameters()
            objp = np.zeros((11 * 8, 3), np.float32)
            objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
            ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
            rvecs = np.float32(rvecs)
            tvecs = np.float32(tvecs)
            x = [[7,5,0],[4,5,0],[1,5,0],[7,2,0],[4,2,0],[1,2,0]]
            
            for j in range(len(word_arr)):
                line = np.float32(word_arr[j] + x[j]).reshape(-1, 3)
                imgpts, jac = cv2.projectPoints(line, rvecs, tvecs, mtx, dist)
                img = self.draw(img, corners2, imgpts, word_point[j])
            img = cv2.resize(img,  (1024, 1024))
            cv2.imshow("windows_name", img)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()

    def on_btn3_1_click(self):
        imgL = cv2.imread('./Dataset_CvDl_Hw1/Q3_Image/imL.png', cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread('./Dataset_CvDl_Hw1/Q3_Image/imR.png', cv2.IMREAD_GRAYSCALE)
        stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
        disparity = stereo.compute(imgL, imgR)
        disparity = cv2.normalize(disparity, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        height,weight = disparity.shape[0],disparity.shape[1]
        print(height,weight)
        dis = cv2.resize(disparity, (1000, 680))
        

        def onclick(event):
            circle1 = plt.Circle((event.xdata,event.ydata), 10, color='green')
            f = 4019   # pixels
            B = 342    # mm
            d = disparity[int(event.ydata)][int(event.xdata)]
            
            x = event.xdata - d
            y = event.ydata
            circle2 = plt.Circle((x, y), 10, color='green')
            ax1.add_patch(circle1)
            ax2.add_patch(circle2)
            figL.canvas.draw()
            figR.canvas.draw()

            
            print('Disparity: ' + str(d) + ' pixels\n' + 'Depth: ' + str(int(f * B / (d + 123))) + ' mm')

        imgL = cv2.imread('./Dataset_CvDl_Hw1/Q3_Image/imL.png')
        imgR = cv2.imread('./Dataset_CvDl_Hw1/Q3_Image/imR.png')

        #imgL = cv2.resize(self.img21, (1000, 680))
        figL = plt.figure(figsize=(12, 8)) 
        imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
        ax1 = figL.add_subplot(111)
        plt.axis('off')
        plt.imshow(imgL)

        #imgR = cv2.resize(self.img22, (1000, 680))
        figR = plt.figure(figsize=(12, 8))
        imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)
        ax2 = figR.add_subplot(111)
        plt.axis('off')
        plt.imshow(imgR)

        figG = plt.figure(figsize=(12, 8))
        ax3 = figG.add_subplot(111)
        plt.axis('off')
        plt.imshow(dis, 'gray')

        cid = figL.canvas.mpl_connect('button_press_event', onclick)
        plt.axis('off')
        plt.show()

    def on_btn4_1_click(self):
        gray_1 = cv2.cvtColor(self.img23, cv2.COLOR_BGR2GRAY)
        gray_2 = cv2.cvtColor(self.img24, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp1 = sift.detect(gray_1,None)
        kp2 = sift.detect(gray_2,None)
        kp1 = sorted(kp1,key = lambda x:x.size,reverse = True)[:200]
        kp2 = sorted(kp2,key =lambda x:x.size,reverse = True)[:200]
        print(len(kp1),len(kp2))
        imgL=cv2.drawKeypoints(gray_1,kp1,self.img23)
        imgR=cv2.drawKeypoints(gray_2,kp2,self.img24)
        horizontal = np.hstack((imgL,imgR))
        cv2.imshow('FeatureShark1',horizontal)
        #cv2.imshow('FeatureShark2',imgR)
        cv2.waitKey(0)

    def on_btn4_2_click(self):
        gray_1 = cv2.cvtColor(self.img23, cv2.COLOR_BGR2GRAY)
        gray_2 = cv2.cvtColor(self.img24, cv2.COLOR_BGR2GRAY)
        # Initiate ORB detector
        orb = cv2.ORB_create()
        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(gray_1,None)
        kp2, des2 = orb.detectAndCompute(gray_2,None)
        print(len(des1[0]))
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1,des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        # Draw first 10 matches.
        img3 = cv2.drawMatches(gray_1,kp1,gray_2,kp2,matches[:200],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.figure(figsize=(12, 8)) 
        plt.axis('off')
        plt.imshow(img3)
        plt.show()

    def on_btn4_3_click(self):
        gray_1 = cv2.cvtColor(self.img23, cv2.COLOR_BGR2GRAY)
        gray_2 = cv2.cvtColor(self.img24, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray_1,None)
        kp2, des2 = sift.detectAndCompute(gray_2,None)

        matcher = cv2.BFMatcher()
        matches = matcher.match(des1,des2)
        matches = sorted(matches, key = lambda x:x.distance)[:200]
        obj = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        scene = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        
        M, Mask = cv2.findHomography(obj,scene,cv2.RANSAC,5.0)
        a = cv2.warpPerspective(gray_2, M, (2*gray_1.shape[0], gray_1.shape[1]))
        a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 8)) 
        print("GG")
        plt.axis('off')
        plt.imshow(a)
        plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())