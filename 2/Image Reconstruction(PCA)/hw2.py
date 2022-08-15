import sys
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox
from numpy.lib.type_check import imag
from hw2_ui import Ui_MainWindow
import glob
from sklearn.decomposition import PCA
from scipy.stats import stats
import matplotlib.image as mpimg
import pandas as pd
from PIL import Image

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.BindComponents()

    def BindComponents(self):
        self.btn4_1.clicked.connect(self.btn4_1_click)
        self.btn4_2.clicked.connect(self.btn4_2_click)

    def btn4_1_click(self):
        file = glob.glob('./Dataset_CvDl_Hw2/Q4_Image/*.jpg')
        fig = plt.figure(figsize = (16, 9))
        for i in range(len(file)):
            #image = cv2.imread(file[i])
            image = cv2.cvtColor(cv2.imread(file[i]), cv2.COLOR_BGR2RGB)
            blue,green,red = cv2.split(image)
            b_df = pd.DataFrame(data = blue)
            r_df = pd.DataFrame(data = red)
            g_df = pd.DataFrame(data = green)

            df_blue = blue/255
            df_green = green/255
            df_red = red/255

            pca_b = PCA(n_components=10)
            pca_b.fit(df_blue)
            trans_pca_b = pca_b.transform(df_blue)
            pca_g = PCA(n_components=10)
            pca_g.fit(df_green)
            trans_pca_g = pca_g.transform(df_green)
            pca_r = PCA(n_components=10)
            pca_r.fit(df_red)
            trans_pca_r = pca_r.transform(df_red)

            b_arr = pca_b.inverse_transform(trans_pca_b)
            g_arr = pca_g.inverse_transform(trans_pca_g)
            r_arr = pca_r.inverse_transform(trans_pca_r)
            img_reduced= (cv2.merge((b_arr, g_arr, r_arr)))
            img_reduced *= 255
            cv2.imwrite('./Dataset_CvDl_Hw2/Q4_1_Image/'+str(i+1)+'.jpg', img_reduced)

            if i < 15:
                fig.add_subplot(4,15,i+1)
                plt.axis('off')
                plt.title(str(i+1))
                plt.imshow(image)

                fig.add_subplot(4,15,i+16)
                plt.axis('off')
                plt.title(str(i+1))
                plt.imshow(img_reduced)
            else:
                fig.add_subplot(4,15,i+16)
                plt.axis('off')
                plt.title(str(i+1))
                plt.imshow(image)

                fig.add_subplot(4,15,i+31)
                plt.axis('off')
                plt.title(str(i+1))
                plt.imshow(img_reduced)
            # cv2.imshow(str(i),reconstruct)
            # cv2.waitKey(500)
            # cv2.destroyAllWindows()
        plt.show()

    def btn4_2_click(self):
        file = glob.glob('./Dataset_CvDl_Hw2/Q4_Image/*.jpg')
        re_file = glob.glob('./Dataset_CvDl_Hw2/Q4_1_Image/*.jpg')
        answer = []
        for i in range(len(file)):
            #image = cv2.imread(file[i])
            img = Image.open(file[i])
            img = img.convert('L')
            re_img = Image.open(re_file[i])
            re_img = re_img.convert('L')
            img_pix = img.load()
            re_pix = re_img.load()
            pixel_value = 0
            for y in range(img.size[0]-1):
                for x in range(img.size[1]-1):
                    temp = abs(img_pix[x,y] - re_pix[x,y])
                    pixel_value += temp
            
            print(pixel_value)
            answer.append(pixel_value)
            
        print(answer)
                    





if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())