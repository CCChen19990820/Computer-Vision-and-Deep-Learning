# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hw2.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(341, 471)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_5.setGeometry(QtCore.QRect(50, 10, 251, 401))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.groupBox_5.setFont(font)
        self.groupBox_5.setObjectName("groupBox_5")
        self.btn4_1 = QtWidgets.QPushButton(self.groupBox_5)
        self.btn4_1.setGeometry(QtCore.QRect(10, 69, 229, 91))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.btn4_1.setFont(font)
        self.btn4_1.setAutoRepeatDelay(-2)
        self.btn4_1.setFlat(False)
        self.btn4_1.setObjectName("btn4_1")
        self.btn4_2 = QtWidgets.QPushButton(self.groupBox_5)
        self.btn4_2.setGeometry(QtCore.QRect(10, 200, 229, 91))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.btn4_2.setFont(font)
        self.btn4_2.setAutoRepeatDelay(-2)
        self.btn4_2.setFlat(False)
        self.btn4_2.setObjectName("btn4_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 341, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox_5.setTitle(_translate("MainWindow", "4. PCA"))
        self.btn4_1.setText(_translate("MainWindow", "4.1 PCA_Reconstruction"))
        self.btn4_2.setText(_translate("MainWindow", "4.2 Reconstruction Error"))

