# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hw2_5ui.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Question_5(object):
    def setupUi(self, Question_5):
        Question_5.setObjectName("Question_5")
        Question_5.resize(276, 304)
        self.centralwidget = QtWidgets.QWidget(Question_5)
        self.centralwidget.setObjectName("centralwidget")
        self.Input_1 = QtWidgets.QSpinBox(self.centralwidget)
        self.Input_1.setGeometry(QtCore.QRect(40, 170, 211, 41))
        self.Input_1.setMaximum(9999)
        self.Input_1.setObjectName("Input_1")
        self.button_1 = QtWidgets.QPushButton(self.centralwidget)
        self.button_1.setGeometry(QtCore.QRect(40, 21, 211, 41))
        self.button_1.setObjectName("button_1")
        self.button_3 = QtWidgets.QPushButton(self.centralwidget)
        self.button_3.setGeometry(QtCore.QRect(40, 120, 211, 41))
        self.button_3.setObjectName("button_3")
        self.button_4 = QtWidgets.QPushButton(self.centralwidget)
        self.button_4.setGeometry(QtCore.QRect(40, 221, 211, 41))
        self.button_4.setObjectName("button_4")
        self.button_2 = QtWidgets.QPushButton(self.centralwidget)
        self.button_2.setGeometry(QtCore.QRect(40, 70, 211, 41))
        self.button_2.setObjectName("button_2")
        Question_5.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Question_5)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 276, 21))
        self.menubar.setObjectName("menubar")
        Question_5.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Question_5)
        self.statusbar.setObjectName("statusbar")
        Question_5.setStatusBar(self.statusbar)

        self.retranslateUi(Question_5)
        QtCore.QMetaObject.connectSlotsByName(Question_5)

    def retranslateUi(self, Question_5):
        _translate = QtCore.QCoreApplication.translate
        Question_5.setWindowTitle(_translate("Question_5", "MainWindow"))
        self.button_1.setText(_translate("Question_5", "1. Show Model Structure"))
        self.button_3.setText(_translate("Question_5", "3.Test"))
        self.button_4.setText(_translate("Question_5", "4. Random-Erasing"))
        self.button_2.setText(_translate("Question_5", "2. Show TensorBoard"))

