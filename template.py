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
        MainWindow.resize(970, 595)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.xAxisLabel = QtWidgets.QLabel(self.centralwidget)
        self.xAxisLabel.setMinimumSize(QtCore.QSize(35, 0))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.xAxisLabel.setFont(font)
        self.xAxisLabel.setObjectName("xAxisLabel")
        self.horizontalLayout_2.addWidget(self.xAxisLabel)
        self.xAxisComboBox = QtWidgets.QComboBox(self.centralwidget)
        self.xAxisComboBox.setMinimumSize(QtCore.QSize(160, 20))
        self.xAxisComboBox.setObjectName("xAxisComboBox")
        self.xAxisComboBox.addItem("")
        self.xAxisComboBox.addItem("")
        self.xAxisComboBox.addItem("")
        self.xAxisComboBox.addItem("")
        self.xAxisComboBox.addItem("")
        self.xAxisComboBox.addItem("")
        self.xAxisComboBox.addItem("")
        self.xAxisComboBox.addItem("")
        self.horizontalLayout_2.addWidget(self.xAxisComboBox)
        self.yAxisLabel = QtWidgets.QLabel(self.centralwidget)
        self.yAxisLabel.setMinimumSize(QtCore.QSize(35, 0))
        self.yAxisLabel.setObjectName("yAxisLabel")
        self.horizontalLayout_2.addWidget(self.yAxisLabel)
        self.yAxisComboBox = QtWidgets.QComboBox(self.centralwidget)
        self.yAxisComboBox.setMinimumSize(QtCore.QSize(160, 20))
        self.yAxisComboBox.setObjectName("yAxisComboBox")
        self.yAxisComboBox.addItem("")
        self.yAxisComboBox.addItem("")
        self.yAxisComboBox.addItem("")
        self.yAxisComboBox.addItem("")
        self.yAxisComboBox.addItem("")
        self.yAxisComboBox.addItem("")
        self.yAxisComboBox.addItem("")
        self.horizontalLayout_2.addWidget(self.yAxisComboBox)
        self.plotPushButton = QtWidgets.QPushButton(self.centralwidget)
        self.plotPushButton.setMinimumSize(QtCore.QSize(70, 0))
        self.plotPushButton.setObjectName("plotPushButton")
        self.horizontalLayout_2.addWidget(self.plotPushButton)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.gridLayout_2.addLayout(self.horizontalLayout_2, 1, 0, 1, 1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout_2.addItem(spacerItem1)
        self.gridLayout_2.addLayout(self.verticalLayout_2, 2, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.dataAnalysisLabel = QtWidgets.QLabel(self.centralwidget)
        self.dataAnalysisLabel.setObjectName("dataAnalysisLabel")
        self.horizontalLayout.addWidget(self.dataAnalysisLabel)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.gridLayout_2.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 970, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuView.setObjectName("menuView")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        self.menuretrian = QtWidgets.QMenu(self.menubar)
        self.menuretrian.setObjectName("menuretrian")
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
        self.predictionDockWidget.setMinimumSize(QtCore.QSize(316, 330))
        self.predictionDockWidget.setObjectName("predictionDockWidget")
        self.predictionDockWidgetContents = QtWidgets.QWidget()
        self.predictionDockWidgetContents.setObjectName("predictionDockWidgetContents")
        self.layoutWidget = QtWidgets.QWidget(self.predictionDockWidgetContents)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 10, 291, 291))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.autofillPushButton = QtWidgets.QPushButton(self.layoutWidget)
        self.autofillPushButton.setObjectName("autofillPushButton")
        self.gridLayout.addWidget(self.autofillPushButton, 2, 0, 1, 1)
        self.predictPushButton = QtWidgets.QPushButton(self.layoutWidget)
        self.predictPushButton.setObjectName("predictPushButton")
        self.gridLayout.addWidget(self.predictPushButton, 2, 1, 1, 1)
        self.predictionLabel = QtWidgets.QLabel(self.layoutWidget)
        self.predictionLabel.setObjectName("predictionLabel")
        self.gridLayout.addWidget(self.predictionLabel, 0, 0, 1, 1)
        self.predictionTableWidget = QtWidgets.QTableWidget(self.layoutWidget)
        self.predictionTableWidget.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.DefaultContextMenu)
        self.predictionTableWidget.setAutoFillBackground(True)
        self.predictionTableWidget.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhPreferNumbers)
        self.predictionTableWidget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.predictionTableWidget.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.predictionTableWidget.setAutoScroll(True)
        self.predictionTableWidget.setProperty("showDropIndicator", True)
        self.predictionTableWidget.setDragDropOverwriteMode(True)
        self.predictionTableWidget.setAlternatingRowColors(True)
        self.predictionTableWidget.setGridStyle(QtCore.Qt.PenStyle.SolidLine)
        self.predictionTableWidget.setObjectName("predictionTableWidget")
        self.predictionTableWidget.setColumnCount(1)
        self.predictionTableWidget.setRowCount(10)
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
        self.predictionTableWidget.setVerticalHeaderItem(9, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.predictionTableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.predictionTableWidget.setItem(0, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.predictionTableWidget.setItem(1, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.predictionTableWidget.setItem(2, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.predictionTableWidget.setItem(3, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.predictionTableWidget.setItem(4, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.predictionTableWidget.setItem(5, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.predictionTableWidget.setItem(6, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.predictionTableWidget.setItem(7, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.predictionTableWidget.setItem(8, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.predictionTableWidget.setItem(9, 0, item)
        self.predictionTableWidget.horizontalHeader().setVisible(True)
        self.predictionTableWidget.horizontalHeader().setCascadingSectionResizes(False)
        self.predictionTableWidget.horizontalHeader().setSortIndicatorShown(False)
        self.predictionTableWidget.horizontalHeader().setStretchLastSection(True)
        self.predictionTableWidget.verticalHeader().setVisible(True)
        self.predictionTableWidget.verticalHeader().setCascadingSectionResizes(False)
        self.predictionTableWidget.verticalHeader().setHighlightSections(True)
        self.predictionTableWidget.verticalHeader().setSortIndicatorShown(False)
        self.predictionTableWidget.verticalHeader().setStretchLastSection(True)
        self.gridLayout.addWidget(self.predictionTableWidget, 1, 0, 1, 2)
        self.predictionDockWidget.setWidget(self.predictionDockWidgetContents)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.predictionDockWidget)
        self.resultDockWidget = QtWidgets.QDockWidget(MainWindow)
        self.resultDockWidget.setMinimumSize(QtCore.QSize(316, 130))
        self.resultDockWidget.setObjectName("resultDockWidget")
        self.resultDockWidgetContents = QtWidgets.QWidget()
        self.resultDockWidgetContents.setObjectName("resultDockWidgetContents")
        self.resultSplitter = QtWidgets.QSplitter(self.resultDockWidgetContents)
        self.resultSplitter.setGeometry(QtCore.QRect(10, 10, 291, 91))
        self.resultSplitter.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.resultSplitter.setObjectName("resultSplitter")
        self.resultLabel = QtWidgets.QLabel(self.resultSplitter)
        self.resultLabel.setObjectName("resultLabel")
        self.resultOutput = QtWidgets.QTextEdit(self.resultSplitter)
        self.resultOutput.setObjectName("resultOutput")
        self.resultDockWidget.setWidget(self.resultDockWidgetContents)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.resultDockWidget)
        self.actionAbout = QtGui.QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.actionImport_widget = QtGui.QAction(MainWindow)
        self.actionImport_widget.setObjectName("actionImport_widget")
        self.actionPrediction_widget = QtGui.QAction(MainWindow)
        self.actionPrediction_widget.setObjectName("actionPrediction_widget")
        self.actionResult_widget = QtGui.QAction(MainWindow)
        self.actionResult_widget.setObjectName("actionResult_widget")
        self.actionPredict = QtGui.QAction(MainWindow)
        self.actionPredict.setObjectName("actionPredict")
        self.actionClose = QtGui.QAction(MainWindow)
        self.actionClose.setObjectName("actionClose")
        self.actionTraining_data = QtGui.QAction(MainWindow)
        self.actionTraining_data.setObjectName("actionTraining_data")
        self.actionAnalysis_data = QtGui.QAction(MainWindow)
        self.actionAnalysis_data.setObjectName("actionAnalysis_data")
        self.actionRe_train = QtGui.QAction(MainWindow)
        self.actionRe_train.setObjectName("actionRe_train")
        self.actionImport_data = QtGui.QAction(MainWindow)
        self.actionImport_data.setObjectName("actionImport_data")
        self.menuFile.addAction(self.actionImport_data)
        self.menuFile.addAction(self.actionPredict)
        self.menuFile.addAction(self.actionClose)
        self.menuView.addAction(self.actionImport_widget)
        self.menuView.addAction(self.actionPrediction_widget)
        self.menuView.addAction(self.actionResult_widget)
        self.menuHelp.addAction(self.actionAbout)
        self.menuretrian.addAction(self.actionRe_train)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.menubar.addAction(self.menuretrian.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.xAxisLabel.setText(_translate("MainWindow", "X-Axis:"))
        self.xAxisComboBox.setItemText(0, _translate("MainWindow", "Dissolved Oxygen (mg/l)"))
        self.xAxisComboBox.setItemText(1, _translate("MainWindow", "pH"))
        self.xAxisComboBox.setItemText(2, _translate("MainWindow", "Conductivity (µmhos/cm)"))
        self.xAxisComboBox.setItemText(3, _translate("MainWindow", "B.O.D. (mg/l)"))
        self.xAxisComboBox.setItemText(4, _translate("MainWindow", "Nitrate (mg/l)"))
        self.xAxisComboBox.setItemText(5, _translate("MainWindow", "Fecal Coliform (MPN/100ml)"))
        self.xAxisComboBox.setItemText(6, _translate("MainWindow", "Total Coliform (MPN/100ml)"))
        self.xAxisComboBox.setItemText(7, _translate("MainWindow", "Years"))
        self.yAxisLabel.setText(_translate("MainWindow", "Y-Axis:"))
        self.yAxisComboBox.setItemText(0, _translate("MainWindow", "Dissolved Oxygen (mg/l)"))
        self.yAxisComboBox.setItemText(1, _translate("MainWindow", "pH"))
        self.yAxisComboBox.setItemText(2, _translate("MainWindow", "Conductivity (µmhos/cm)"))
        self.yAxisComboBox.setItemText(3, _translate("MainWindow", "B.O.D. (mg/l)"))
        self.yAxisComboBox.setItemText(4, _translate("MainWindow", "Nitrate (mg/l)"))
        self.yAxisComboBox.setItemText(5, _translate("MainWindow", "Fecal Coliform (MPN/100ml)"))
        self.yAxisComboBox.setItemText(6, _translate("MainWindow", "Total Coliform (MPN/100 ml)"))
        self.plotPushButton.setText(_translate("MainWindow", "Plot"))
        self.dataAnalysisLabel.setText(_translate("MainWindow", "Data Analysis"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuView.setTitle(_translate("MainWindow", "View"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.menuretrian.setTitle(_translate("MainWindow", "Model"))
        self.importLabel.setText(_translate("MainWindow", "Import File"))
        self.importPushButton.setText(_translate("MainWindow", "Choose File"))
        self.autofillPushButton.setText(_translate("MainWindow", "Autofill"))
        self.predictPushButton.setText(_translate("MainWindow", "Predict"))
        self.predictionLabel.setText(_translate("MainWindow", "Prediction"))
        self.predictionTableWidget.setSortingEnabled(False)
        item = self.predictionTableWidget.verticalHeaderItem(0)
        item.setText(_translate("MainWindow", "WQI 1 year"))
        item = self.predictionTableWidget.verticalHeaderItem(1)
        item.setText(_translate("MainWindow", "WQI 2 years"))
        item = self.predictionTableWidget.verticalHeaderItem(2)
        item.setText(_translate("MainWindow", "WQI 3 years"))
        item = self.predictionTableWidget.verticalHeaderItem(3)
        item.setText(_translate("MainWindow", "Dissolved Oxygen (mg/l)"))
        item = self.predictionTableWidget.verticalHeaderItem(4)
        item.setText(_translate("MainWindow", "pH"))
        item = self.predictionTableWidget.verticalHeaderItem(5)
        item.setText(_translate("MainWindow", "Conductivity (µmhos/cm)"))
        item = self.predictionTableWidget.verticalHeaderItem(6)
        item.setText(_translate("MainWindow", "B.O.D. (mg/l)"))
        item = self.predictionTableWidget.verticalHeaderItem(7)
        item.setText(_translate("MainWindow", "Nitrate (mg/l)"))
        item = self.predictionTableWidget.verticalHeaderItem(8)
        item.setText(_translate("MainWindow", "Fecal Coliform (MPN/100ml)"))
        item = self.predictionTableWidget.verticalHeaderItem(9)
        item.setText(_translate("MainWindow", "Total Coliform (MPN/100ml)"))
        item = self.predictionTableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Value"))
        __sortingEnabled = self.predictionTableWidget.isSortingEnabled()
        self.predictionTableWidget.setSortingEnabled(False)
        self.predictionTableWidget.setSortingEnabled(__sortingEnabled)
        self.resultLabel.setText(_translate("MainWindow", "Result"))
        self.resultOutput.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.actionAbout.setText(_translate("MainWindow", "About"))
        self.actionImport_widget.setText(_translate("MainWindow", "Import widget"))
        self.actionPrediction_widget.setText(_translate("MainWindow", "Prediction widget"))
        self.actionResult_widget.setText(_translate("MainWindow", "Result widget"))
        self.actionPredict.setText(_translate("MainWindow", "Predict"))
        self.actionClose.setText(_translate("MainWindow", "Close"))
        self.actionTraining_data.setText(_translate("MainWindow", "Training data"))
        self.actionAnalysis_data.setText(_translate("MainWindow", "Analysis data"))
        self.actionRe_train.setText(_translate("MainWindow", "Re-train"))
        self.actionImport_data.setText(_translate("MainWindow", "Import data"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())
