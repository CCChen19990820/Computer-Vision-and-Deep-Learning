import sys
import cv2
import numpy as np
import random
from scipy import signal
import matplotlib.pyplot as plt
import keras
from keras import optimizers, regularizers
from keras import backend as K
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from keras.utils import np_utils
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.layers.core import Lambda
from hw1_5_ui import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox



class MyVGG16():
    def __init__(self):
        self.weight_decay = 0.0005
        self.nb_epoch = 20
        self.batch_size = 32
        self.learning_rate = 0.001
        self.optimizer = 'SGD'
    
    def BuildModel(self):
        model = Sequential()
        #layer1 32*32*3
        model.add(Conv2D(64, (3, 3), padding='same', input_shape=(32,32,3), kernel_regularizer=regularizers.l2(self.weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        #layer2 32*32*64
        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #layer3 16*16*64
        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        #layer4 16*16*128
        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #layer5 8*8*128
        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        #layer6 8*8*256
        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        #layer7 8*8*256
        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #layer8 4*4*256
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        #layer9 4*4*512
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        #layer10 4*4*512
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #layer11 2*2*512
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        #layer12 2*2*512
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        #layer13 2*2*512
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        #layer14 1*1*512
        model.add(Flatten())
        model.add(Dense(512, kernel_regularizer=regularizers.l2(self.weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        #layer15 512
        model.add(Dense(512, kernel_regularizer=regularizers.l2(self.weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        #layer16 512
        model.add(Dropout(0.5))
        model.add(Dense(10))
        model.add(Activation('softmax'))
        sgd = optimizers.gradient_descent_v2.SGD(lr=self.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return model

    def TrainModel(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        y_train = np_utils.to_categorical(y_train, 10)
        y_test = np_utils.to_categorical(y_test, 10)
        model = self.BuildModel()
        history = model.fit(x_train, y_train, epochs=self.nb_epoch, batch_size=self.batch_size, validation_split=0.1, verbose=1)
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

    def Get(self):
        return self.batch_size, self.learning_rate, self.optimizer

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.BindComponents()
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
        self.labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.y_train = np_utils.to_categorical(self.y_train, 10)
        self.y_test = np_utils.to_categorical(self.y_test, 10)
        self.myVGG16 = MyVGG16()
        self.model = load_model('model.h5')
        '''
        try:
            self.model = load_model('model.h5')
        except:
            self.model = self.myVGG16.TrainModel()
        '''


    def BindComponents(self):
        self.button_1.clicked.connect(self.on_button_1_click)
        self.button_2.clicked.connect(self.on_button_2_click)
        self.button_3.clicked.connect(self.on_button_3_click)
        self.button_4.clicked.connect(self.on_button_4_click)
        self.button_5.clicked.connect(self.on_button_5_click)

    def on_button_1_click(self):
        x = np.concatenate((self.x_train, self.x_test), axis=0)
        y = np.concatenate((self.y_train, self.y_test), axis=0)
        fig = plt.figure(figsize=(18, 9))
        arr = []
        for i in range(1, 10):
            n = random.randint(0, len(y))
            while n in arr:
                n = random.randint(0, len(y))
            arr.append(n)
            fig.add_subplot(3, 3, i)
            plt.imshow(x[n].astype(np.uint8), interpolation='nearest')
            plt.axis('off')
            label = list(y[n]).index(max(y[n]))
            plt.text(0, 0, self.labels[label], fontsize=12, verticalalignment="bottom")
        plt.show()


    def on_button_2_click(self):

        batch_size, learning_rate, optimizer = self.myVGG16.Get()
        print('hyperparameters:')
        print('batch size: ' + str(batch_size))
        print('learning rate: ' + str(learning_rate))
        print('optimizer: ' + optimizer)

    def on_button_3_click(self):
        print(self.model.summary())

    def on_button_4_click(self):
        img = cv2.imread('accuracy.png')
        img2 = cv2.imread('loss.png')
        cv2.imshow('Accuracy', img)
        cv2.imshow('Loss', img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_button_5_click(self):
        n = self.spin_1.value()
        x = np.copy(self.x_test)
        img = x[n]
        #plt.imshow(img)
        #plt.show()
        img = np.expand_dims(img, axis=0)
        predict = self.model.predict(img)
        print(predict)

        #Left Image
        fig = plt.figure(figsize=(21, 9))
        fig.add_subplot(1, 2, 1)
        plt.axis('off')
        plt.title('Test Image')
        plt.imshow(self.x_test[n].astype(np.uint8), interpolation='nearest')

        #Right Table
        label = list(self.y_test[n]).index(max(self.y_test[n]))
        plt.text(0, -1, self.labels[label], fontsize=12, verticalalignment="bottom")
        fig.add_subplot(1, 2, 2)
        plt.title('Classification probability distribution table')
        
        plt.bar(self.labels, predict[0])
        plt.ylabel('Probability')
        plt.xlabel('Label')
        plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
