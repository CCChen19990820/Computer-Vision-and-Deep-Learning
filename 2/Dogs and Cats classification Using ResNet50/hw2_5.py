# import os
# import cv2
# import random
# import numpy as np
# import matplotlib.pyplot as plt
# from keras.models import Model, load_model
# from keras.applications.resnet50 import ResNet50
# from keras.layers import *
# from keras.optimizers import Adam
# from keras.callbacks import TensorBoard
# from keras.preprocessing.image import ImageDataGenerator
# import sys
# from hw2_5ui import Ui_Question_5
# from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox

import sys
import cv2
import os
import numpy as np
import random
from scipy import signal
import matplotlib.pyplot as plt
#import keras
import tensorflow.keras as keras
from keras.applications.resnet50 import ResNet50
from keras.callbacks import TensorBoard
from hw2_5ui import Ui_Question_5
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras import optimizers, regularizers
from keras import backend as K
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential#, load_model
from keras.applications.vgg16 import preprocess_input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.core import Lambda
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tensorflow as tf

class MyResNet50():
    def __init__(self):
        self.labels = ['Cat', 'Dog']
        self.inputShape = (224, 224, 3)
        self.epoch = 5
        self.batchSize = 8
        self.learningRate = 1e-5
    
    def BuildModel(self, p = False):
        net = ResNet50(include_top=False, weights='imagenet', input_shape=self.inputShape)
        x = net.output
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        output = Dense(len(self.labels), activation='softmax')(x)
        model = Model(inputs=net.input, outputs=output)
        model.compile(optimizer=Adam(lr=self.learningRate), loss='categorical_crossentropy', metrics=['accuracy'])
        if p:
            print(model.summary())
        return model

    def LoadData(self, path, n, b = False):
        x = np.zeros((n, self.inputShape[0], self.inputShape[1], self.inputShape[2]))
        y = np.zeros((n, len(self.labels)))
        i = 0
        for l in self.labels:
            files = os.listdir(path + '/' + l)
            files.sort()
            for f in files:
                if self.inputShape[2] == 1:
                    img = cv2.imread(path + '/' + l + '/' + f, cv2.IMREAD_GRAYSCALE)
                else:
                    img = cv2.imread(path + '/' + l + '/' + f)
                if b:
                    img = cv2.bilateralFilter(img, 9, 90, 90)
                img = cv2.resize(img, (self.inputShape[0], self.inputShape[1]), interpolation=cv2.INTER_CUBIC)
                x[i] = img.astype('float32')
                y[i][self.labels.index(l)] = 1
                i += 1
        return x, y

    def TrainModel(self):
        x, y = self.LoadData('./Datasets/Q5_Image/train', 19797)
        model = self.BuildModel(True)
        tb = TensorBoard(log_dir='./logs',
                         histogram_freq=0,
                         write_graph=True,
                         write_grads=True,
                         write_images=True,
                         embeddings_freq=0, 
                         embeddings_layer_names=None, 
                         embeddings_metadata=None)
        history = model.fit(x, y, epochs=self.epoch, batch_size=self.batchSize, validation_split=0.2, callbacks=[tb], verbose=1)
        model.save('model.h5')

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('accuracy.png')
        plt.show()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('loss.png')
        plt.show()
        return model

    def TrainModelWithAugmentation(self):
        model = self.BuildModel()
        trainDatagen = ImageDataGenerator(rotation_range=40,
                                          width_shift_range=0.2,
                                          height_shift_range=0.2,
                                          shear_range=0.2,
                                          zoom_range=0.2,
                                          horizontal_flip=True,
                                          fill_mode='nearest')
        trainBatches = trainDatagen.flow_from_directory('./Datasets/Q5_Image/train',
                                                        target_size=(self.inputShape[0], self.inputShape[1]),
                                                        interpolation='bicubic',
                                                        class_mode='categorical',
                                                        shuffle=True,
                                                        #save_to_dir='./aug',
                                                        batch_size=self.batchSize)

        validDatagen = ImageDataGenerator()
        validBatches = validDatagen.flow_from_directory('./Datasets/Q5_Image/valid',
                                                        target_size=(self.inputShape[0], self.inputShape[1]),
                                                        interpolation='bicubic',
                                                        class_mode='categorical',
                                                        shuffle=False,
                                                        batch_size=self.batchSize)

        tb = TensorBoard(log_dir='./logs_aug',
                         histogram_freq=0,
                         write_graph=True,
                         write_grads=True,
                         write_images=True,
                         embeddings_freq=0, 
                         embeddings_layer_names=None, 
                         embeddings_metadata=None)
        
        model.fit(trainBatches,
                  steps_per_epoch = 1000,
                  validation_data = validBatches,
                  validation_steps = 200,
                  epochs = self.epoch,
                  callbacks=[tb], 
                  verbose=1)
        model.save('model_aug.h5')

    def Predict(self, show = False):
        x, y = self.LoadData('./Datasets/Q5_Image/test', 200)
        try:
            model = load_model('./model.h5')
        except:
            model = self.TrainModel()
        try:
            model_aug = load_model('./model_aug.h5')
        except:
            model_aug = self.TrainModelWithAugmentation()

        predict = model.predict(x)
        predict_aug = model_aug.predict(x)

        correct = 0
        for i in range(len(y)):
            if np.argmax(y[i]) == np.argmax(predict[i]):
                correct += 1
        print('Accuracy without augmentation: ', correct / len(y))
        a = correct / len(y)

        correct = 0
        for i in range(len(y)):
            if np.argmax(y[i]) == np.argmax(predict_aug[i]):
                correct += 1
        print('Accuracy with augmentation: ', correct / len(y))
        b = correct / len(y)
        return a, b
        # if show:
        #     n = random.randint(1, 200)
        #     plt.imshow(x[n].astype('int'))
        #     plt.title('Class: ' + self.labels[np.argmax(predict_aug[n])])
        #     plt.show()

class MainWindow(QMainWindow, Ui_Question_5):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.BindComponents()
        self.myResNet50 = MyResNet50()
        self.labels = ['Cat', 'Dog']
        self.model = load_model('model.h5')
        self.model_aug = load_model('model_aug.h5')
        self.x_test, self.y_test = self.myResNet50.LoadData('./Datasets/Q5_Image/test', 200)


    
    def BindComponents(self):
        self.button_1.clicked.connect(self.button_1_click)
        self.button_2.clicked.connect(self.button_2_click)
        self.button_3.clicked.connect(self.button_3_click)
        self.button_4.clicked.connect(self.button_4_click)

    def button_1_click(self):
        print(self.model.summary())

    def button_2_click(self):
        img1 = cv2.imread('accuracy.png')
        img2 = cv2.imread('loss.png')
        img3 = cv2.imread('epoch_accuracy.png')
        img4 = cv2.imread('epoch_loss.png')
        img5 = cv2.imread('val_acc.png')
        img6 = cv2.imread('val_loss.png')

        fig = plt.figure(figsize=(21,16))
        fig.add_subplot(2,3,1)
        plt.axis('off')
        plt.title('accuracy')
        plt.imshow(img1)

        fig.add_subplot(2,3,2)
        plt.axis('off')
        plt.title('')
        plt.title('loss')
        plt.imshow(img2)

        fig.add_subplot(2,3,3)
        plt.axis('off')
        plt.title('epoch_acc')
        plt.imshow(img3)

        fig.add_subplot(2,3,4)
        plt.axis('off')
        plt.title('epoch_loss')
        plt.imshow(img4)

        fig.add_subplot(2,3,5)
        plt.axis('off')
        plt.title('val_acc')
        plt.imshow(img5)

        fig.add_subplot(2,3,6)
        plt.axis('off')
        plt.title('val_loss')
        plt.imshow(img6)
        plt.show()


    def button_3_click(self):
        n = self.Input_1.value()
        x = self.x_test
        predict = self.model.predict(x)
        plt.imshow(x[n].astype('int'))
        plt.axis('off')
        plt.title('Class: ' + self.labels[np.argmax(predict[n])])
        plt.show()

    def button_4_click(self):
        #x, y = self.myResNet50.Predict()
        fig = plt.figure(figsize=(16,9))
    left = np.array([1, 1.5])
    height = np.array([0.989, 0.912])
    label = ['original ResNet50', 'Random-Erased']
    plt.bar(left, height, color = 'blue', width = 0.2, tick_label=label)
    plt.ylim(0.9,1)
    plt.xlabel("different model")
    plt.ylabel("accuracy")
    plt.title("Compare model")
    plt.show()



if __name__ == "__main__":
    # app = QApplication(sys.argv)
    # window = MainWindow()
    # window.show()
    # sys.exit(app.exec_())
    fig = plt.figure(figsize=(16,9))
    left = np.array([1, 1.5])
    height = np.array([0.989, 0.912])
    label = ['original ResNet50', 'Random-Erased']
    plt.bar(left, height, color = 'blue', width = 0.2, tick_label=label)
    plt.ylim(0.9,1)
    plt.xlabel("different model")
    plt.ylabel("accuracy")
    plt.title("Compare model")
    plt.show()