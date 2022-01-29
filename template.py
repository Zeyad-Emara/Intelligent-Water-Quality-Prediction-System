# Form implementation generated from reading ui file 'template.ui'
#
# Created by: PyQt6 UI code generator 6.2.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(940, 630)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.dataAnalysisLabel = QtWidgets.QLabel(self.centralwidget)
        self.dataAnalysisLabel.setGeometry(QtCore.QRect(10, 0, 81, 16))
        self.dataAnalysisLabel.setObjectName("dataAnalysisLabel")
        self.graphWidget = QtWidgets.QWidget(self.centralwidget)
        self.graphWidget.setGeometry(QtCore.QRect(10, 20, 571, 561))
        self.graphWidget.setObjectName("graphWidget")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 940, 21))
        self.menubar.setObjectName("menubar")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.importDockWidget = QtWidgets.QDockWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.importDockWidget.sizePolicy().hasHeightForWidth())
        self.importDockWidget.setSizePolicy(sizePolicy)
        self.importDockWidget.setMinimumSize(QtCore.QSize(316, 86))
        self.importDockWidget.setBaseSize(QtCore.QSize(0, 0))
        self.importDockWidget.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.importDockWidget.setObjectName("importDockWidget")
        self.importDockWidgetContents = QtWidgets.QWidget()
        self.importDockWidgetContents.setObjectName("importDockWidgetContents")
        self.importSplitter = QtWidgets.QSplitter(self.importDockWidgetContents)
        self.importSplitter.setGeometry(QtCore.QRect(10, 10, 161, 51))
        self.importSplitter.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.importSplitter.setObjectName("importSplitter")
        self.importLabel = QtWidgets.QLabel(self.importSplitter)
        self.importLabel.setObjectName("importLabel")
        self.importPushButton = QtWidgets.QPushButton(self.importSplitter)
        self.importPushButton.setObjectName("importPushButton")
        self.importDockWidget.setWidget(self.importDockWidgetContents)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.importDockWidget)
        self.predictionDockWidget = QtWidgets.QDockWidget(MainWindow)
        self.predictionDockWidget.setMinimumSize(QtCore.QSize(316, 322))
        self.predictionDockWidget.setObjectName("predictionDockWidget")
        self.predictionDockWidgetContents = QtWidgets.QWidget()
        self.predictionDockWidgetContents.setObjectName("predictionDockWidgetContents")
        self.layoutWidget = QtWidgets.QWidget(self.predictionDockWidgetContents)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 10, 291, 281))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.predictionLabel = QtWidgets.QLabel(self.layoutWidget)
        self.predictionLabel.setObjectName("predictionLabel")
        self.gridLayout.addWidget(self.predictionLabel, 0, 0, 1, 1)
        self.predictionTableWidget = QtWidgets.QTableWidget(self.layoutWidget)
        self.predictionTableWidget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.predictionTableWidget.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.predictionTableWidget.setAutoScroll(True)
        self.predictionTableWidget.setAlternatingRowColors(True)
        self.predictionTableWidget.setGridStyle(QtCore.Qt.PenStyle.SolidLine)
        self.predictionTableWidget.setObjectName("predictionTableWidget")
        self.predictionTableWidget.setColumnCount(1)
        self.predictionTableWidget.setRowCount(9)
        item = QtWidgets.QTableWidgetItem()
        self.predictionTableWidget.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.predictionTableWidget.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.predictionTableWidget.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.predictionTableWidget.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.predictionTableWidget.setVerticalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.predictionTableWidget.setVerticalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.predictionTableWidget.setVerticalHeaderItem(6, item)
        item = QtWidgets.QTableWidgetItem()
        self.predictionTableWidget.setVerticalHeaderItem(7, item)
        item = QtWidgets.QTableWidgetItem()
        self.predictionTableWidget.setVerticalHeaderItem(8, item)
        item = QtWidgets.QTableWidgetItem()
        self.predictionTableWidget.setHorizontalHeaderItem(0, item)
        self.predictionTableWidget.horizontalHeader().setCascadingSectionResizes(False)
        self.gridLayout.addWidget(self.predictionTableWidget, 1, 0, 1, 2)
        self.predictPushButton = QtWidgets.QPushButton(self.layoutWidget)
        self.predictPushButton.setObjectName("predictPushButton")
        self.gridLayout.addWidget(self.predictPushButton, 2, 1, 1, 1)
        self.predictionDockWidget.setWidget(self.predictionDockWidgetContents)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.predictionDockWidget)
        self.resultDockWidget = QtWidgets.QDockWidget(MainWindow)
        self.resultDockWidget.setMinimumSize(QtCore.QSize(316, 173))
        self.resultDockWidget.setObjectName("resultDockWidget")
        self.resultDockWidgetContents = QtWidgets.QWidget()
        self.resultDockWidgetContents.setObjectName("resultDockWidgetContents")
        self.resultSplitter = QtWidgets.QSplitter(self.resultDockWidgetContents)
        self.resultSplitter.setGeometry(QtCore.QRect(10, 10, 291, 141))
        self.resultSplitter.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.resultSplitter.setObjectName("resultSplitter")
        self.resultLabel = QtWidgets.QLabel(self.resultSplitter)
        self.resultLabel.setObjectName("resultLabel")
        self.resultOutput = QtWidgets.QTextEdit(self.resultSplitter)
        self.resultOutput.setObjectName("resultOutput")
        self.resultDockWidget.setWidget(self.resultDockWidgetContents)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.resultDockWidget)
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.dataAnalysisLabel.setText(_translate("MainWindow", "Data Analysis"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.importLabel.setText(_translate("MainWindow", "Import File"))
        self.importPushButton.setText(_translate("MainWindow", "Choose File"))
        self.predictionLabel.setText(_translate("MainWindow", "Prediction"))
        self.predictionTableWidget.setSortingEnabled(False)
        item = self.predictionTableWidget.verticalHeaderItem(0)
        item.setText(_translate("MainWindow", "ph"))
        item = self.predictionTableWidget.verticalHeaderItem(1)
        item.setText(_translate("MainWindow", "Hardness"))
        item = self.predictionTableWidget.verticalHeaderItem(2)
        item.setText(_translate("MainWindow", "Solids"))
        item = self.predictionTableWidget.verticalHeaderItem(3)
        item.setText(_translate("MainWindow", "Chloramines"))
        item = self.predictionTableWidget.verticalHeaderItem(4)
        item.setText(_translate("MainWindow", "Sulfate"))
        item = self.predictionTableWidget.verticalHeaderItem(5)
        item.setText(_translate("MainWindow", "Conductivity"))
        item = self.predictionTableWidget.verticalHeaderItem(6)
        item.setText(_translate("MainWindow", "Organic carbon"))
        item = self.predictionTableWidget.verticalHeaderItem(7)
        item.setText(_translate("MainWindow", "Trihalomethanes"))
        item = self.predictionTableWidget.verticalHeaderItem(8)
        item.setText(_translate("MainWindow", "Turbidity"))
        item = self.predictionTableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Value"))
        self.predictPushButton.setText(_translate("MainWindow", "Predict"))
        self.resultLabel.setText(_translate("MainWindow", "Result"))
        self.resultOutput.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())