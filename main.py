import sys

from PyQt6.QtWidgets import QLineEdit, QItemDelegate, QWidget, QComboBox, QTableView, QStyledItemDelegate, QDialog, \
    QDockWidget

import template
from template import Ui_MainWindow
from about_template import Ui_Dialog as AboutDialog
from PyQt6.QtCore import QRegularExpression, QRect, QCoreApplication
from PyQt6 import QtWidgets as Qtw, QtCore, QtGui, QtWidgets
from PyQt6.QtGui import QDoubleValidator, QValidator, QRegularExpressionValidator, QPen, QColor, QBrush

# for exe file compilation
import sklearn.utils._typedefs
from PyQt6 import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qtagg import (FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from scipy import stats

from joblib import dump, load


class Window(Qtw.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(Window, self).__init__(*args, **kwargs)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.resize(1000, 650)

        # Your code starts here
        self.filePath = None
        self.predict_input = None
        self.predict_value = None
        self.scaler = None
        self.predicting_model = None

        self.has_used_data_for_training = False
        # add default model
        try:
            self.predicting_model = load('resource/models/WQIModelv2_2.pkl')
            self.scaler = load('resource/models/StdScaler_time.pkl')
        except Exception:
            # add error window in following version if needed
            self.ui.statusbar.showMessage("Default modal not detected")
        # add default dataset
        try:
            self.data_frame = pd.read_csv("resource/trainingData/water_dataX_wqi_cleaned.csv")
        except Exception:
            self.ui.statusbar.showMessage("Default dataset not detected")
        self.new_x_axis = 'Dissolved Oxygen (mg/l)'
        self.new_y_axis = 'Dissolved Oxygen (mg/l)'
        # self.isImportDocketClosed = False
        # Update statusbar
        # self.ui.statusbar.showMessage("No dataset found")
        # Draw canvas and Toolbar
        self.layout = Qtw.QVBoxLayout()
        self.static_canvas = FigureCanvasQTAgg(Figure(figsize=(10, 10)))
        self.layout.addWidget(MyToolBar(self.static_canvas, self.centralWidget()))
        self.graph = self.static_canvas.figure.add_subplot(111)
        self.layout.addWidget(self.static_canvas)
        self.ui.verticalLayout_2.addLayout(self.layout)
        # add import button function
        self.ui.importPushButton.clicked.connect(self.find_csv)
        # add predicting value function
        # self.ui.predictPushButton.setEnabled(False)
        self.ui.predictPushButton.clicked.connect(self.prediction)
        # add autofill function
        # self.ui.autofillPushButton.setEnabled(False)
        self.ui.autofillPushButton.clicked.connect(self.auto_fill)
        # set result display area read only
        self.ui.resultOutput.setReadOnly(True)
        # restrict user's input in the table
        self.delegate = TableWidgetDelegate()
        self.ui.predictionTableWidget.setItemDelegateForColumn(0, self.delegate)
        # plot different graph
        self.ui.xAxisComboBox.currentTextChanged.connect(self.change_x_axis)
        self.ui.yAxisComboBox.currentTextChanged.connect(self.change_y_axis)
        self.ui.plotPushButton.clicked.connect(self.change_graph_axis)
        # add import function on menu bar
        self.ui.actionImport_data.triggered.connect(self.find_csv)
        # add redict function on menu bar
        self.ui.actionPredict.triggered.connect(self.prediction)
        # add about page
        self.ui.actionAbout.triggered.connect(self.build_about)
        # add close function
        self.ui.actionClose.triggered.connect(QCoreApplication.instance().quit)
        # add import docker widget function
        self.ui.actionImport_widget.triggered.connect(self.import_docker)
        # add prediction docker widget function
        self.ui.actionPrediction_widget.triggered.connect(self.prediction_docker)
        # add result docker widget function
        self.ui.actionResult_widget.triggered.connect(self.result_docker)
        # add retrain model function
        self.ui.actionRe_train.triggered.connect(self.retrain_model)
        # Your code ends here
        self.show()

    def change_x_axis(self, x_axis):
        self.new_x_axis = x_axis

    def change_y_axis(self, y_axis):
        self.new_y_axis = y_axis

    def change_graph_axis(self):
        column = {'Dissolved Oxygen (mg/l)': 'D.O.',
                  'pH': 'PH',
                  'Conductivity (µmhos/cm)': 'CONDUCTIVITY',
                  'B.O.D. (mg/l)': 'B.O.D.',
                  'Nitrate (mg/l)': 'NITRATE',
                  'Fecal Coliform (MPN/100ml)': 'FECAL COLIFORM',
                  'Total Coliform (MPN/100ml)': 'TOTAL COLIFORM'
                  }
        try:
            self.graph.clear()
            self.data_frame.plot(kind='scatter', x=column[self.new_x_axis], y=column[self.new_y_axis], ax=self.graph)
            self.graph.axes.set_xlabel(self.new_x_axis)
            self.graph.axes.set_ylabel(self.new_y_axis)
            self.static_canvas.draw()
        except Exception as e:
            print(e)

    def find_csv(self):
        try:
            self.filePath = Qtw.QFileDialog.getOpenFileName(filter="csv (*.csv)")[0]
            if self.filePath:
                self.data_frame = pd.read_csv(self.filePath)
                self.ui.statusbar.showMessage("Dataset selected from " + self.filePath)
                self.has_used_data_for_training = False
        except Exception as e:
            print(e)

    def retrain_model(self):

        self.ui.statusbar.showMessage("retraining")

        print("0")

        if self.has_used_data_for_training or self.data_frame is None:
            self.ui.statusbar.showMessage("Data has been used to train the model. Multiple training with same model"
                                          "can overtrain the model")
        # elif self.has_used_data_for_training is None | self.data_frame is None:
        #     print("2")
        #     self.ui.statusbar.showMessage("Please load a dataset to train the model")
        else:
            print("3")
            try:
                training_data = self.preprocess_data()

                x_train = training_data.drop['WQI']
                y_train = training_data['WQI']

                predicting_model.fit(x_train,y_train)

                self.has_used_data_for_training = True

                self.ui.statusbar.showMessage("retraining successful")

                # training_data.to_csv(r'preprocessed_data.csv', index=False, header=True)
            except Exception as e:
                print('training error:'+e)

    def preprocess_data(self):

        data_frame_time = self.data_frame

        #print(self.data_frame)
        #print(data_frame_time)

        try:

            # list of final columns to keep
            final_table_columns = ['year','WQI','D.O.','PH','CONDUCTIVITY','B.O.D.','NITRATE','FECAL COLIFORM','TOTAL COLIFORM']
            # remove unwanted columns
            #data_frame_time = data_frame_time.drop(columns=[col for col in data_frame_time if col not in final_table_columns])
            data_frame_time = data_frame_time.sort_values(by=['year'])
            data_frame_time.dropna()



            dupe = data_frame_time

            # appending the features previous in time
            data_frame_time['WQI t-1'] = dupe['WQI'].shift(1)
            data_frame_time['WQI t-2'] = dupe['WQI'].shift(2)
            data_frame_time['WQI t-3'] = dupe['WQI'].shift(3)
            data_frame_time['D.O. t-1'] = dupe['D.O.'].shift(1)
            data_frame_time['PH t-1'] = dupe['PH'].shift(1)
            data_frame_time['CONDUCTIVITY t-1'] = dupe['CONDUCTIVITY'].shift(1)
            data_frame_time['B.O.D. t-1'] = dupe['B.O.D.'].shift(1)
            data_frame_time['NITRATE t-1'] = dupe['NITRATE'].shift(1)
            data_frame_time['FECAL COLIFORM t-1'] = dupe['FECAL COLIFORM'].shift(1)
            data_frame_time['TOTAL COLIFORM t-1'] = dupe['TOTAL COLIFORM'].shift(1)

            final_table_columns = ['WQI', 'WQI t-1', 'WQI t-2', 'WQI t-3', 'D.O. t-1', 'PH t-1', 'CONDUCTIVITY t-1',
                                   'B.O.D. t-1', 'NITRATE t-1', 'FECAL COLIFORM t-1', 'TOTAL COLIFORM t-1']

            data_frame_time = data_frame_time.dropna()

            data_frame_time = data_frame_time.drop(
                columns=[col for col in data_frame_time if col not in final_table_columns])

            data_frame_time = data_frame_time[(np.abs(stats.zscore(data_frame_time)) < 3).all(axis=1)]

            print(data_frame_time)

            x_time = data_frame_time.drop(columns=['WQI'])

            x_time = self.scaler.transform(x_time)

            y_time = data_frame_time['WQI']

            processed_data = x_time
            processed_data['WQI'] = y_time.tolist()

            return processed_data

        except Exception as e:
            print ('preprocessing error: ' + e)

        return data_frame_time

    def get_feature(self, year, feature):

        water_data = self.data_frame
        try:
            row = water_data.loc[(water_data['year'] == year)]
            return row.iloc[0][feature]
        except:
            return None

    # def plot_graph(self):
    #     try:
    #         self.data_frame = pd.read_csv(self.filePath, encoding='utf-8')
    #         columns = list(self.data_frame.columns)
    #         df = pd.read_csv(self.filePath, usecols=columns)
    #         df.plot(ax=self.graph)
    #         self.graph.legend().set_draggable(True)
    #         self.graph.axes.set_xlabel('X Axis')
    #         self.graph.axes.set_ylabel('Y Axis')
    #         self.graph.axes.set_title('Title')
    #         self.static_canvas.draw()
    #         self.ui.predictPushButton.setEnabled(True)
    #         self.ui.autofillPushButton.setEnabled(True)
    #     except Exception:
    #         self.ui.statusbar.showMessage("Failed to plot graph")

    def prediction(self):

        try:
            value = self.read_table_data()
            if len(value) == 10:
                value = np.array(value).reshape(1, -1)
                self.predict_input = self.scaler.transform(value)
                self.predict_value = self.predicting_model.predict(self.predict_input)
                self.ui.resultOutput.clear()
                self.ui.resultOutput.insertPlainText(str(self.predict_value[0]))
            else:
                self.ui.statusbar.showMessage("Invalid input(s)")
        except Exception as e:
            print(e)
            self.ui.statusbar.showMessage("Something's wrong with the model")

    def read_table_data(self):
        self.ui.statusbar.showMessage("")
        # WQI t-1, WQI t-2, WQI t-3, dissolved oxygen, pH, conductivity, B.O.D., Nitrate, Fecal Coliform, Total Coliform
        item = []
        row_count = self.ui.predictionTableWidget.rowCount()
        # WQI t-1, WQI t-2, WQI t-3, dissolved oxygen, pH, conductivity, B.O.D., Nitrate, Fecal Coliform, Total Coliform
        restriction = [(0, 100), (0, 100), (0, 100), (0, 100), (0, 14), (0, 100), (0, 100), (0, 20), (0, 1000),
                       (0, 1000), ]
        # check for invalid inputs
        for row in range(row_count):
            value = float(self.ui.predictionTableWidget.item(row, 0).text())
            self.ui.predictionTableWidget.item(row, 0).setForeground(QBrush(QColor(0, 0, 0)))
            if round(value) not in range(restriction[row][0], restriction[row][1]):
                self.ui.predictionTableWidget.item(row, 0).setBackground(QColor("red"))
            else:
                self.ui.predictionTableWidget.item(row, 0).setBackground(QColor("white"))
            item.append(value)
        return item

    def auto_fill(self):
        try:
            # column = ['', '', '', 'D.O. (mg/l)', 'PH', 'CONDUCTIVITY (µmhos/cm)', 'B.O.D. (mg/l)',
            #          'NITRATE N+ NITRITEN (mg/l)', 'FECAL COLIFORM (MPN/100ml)', 'TOTAL COLIFORM (MPN/100ml)Mean']
            for idx in range(0, 10):
                if self.ui.predictionTableWidget.item(idx, 0).text() == "":
                    mean = self.scaler.mean_[idx]
                    self.ui.predictionTableWidget.item(idx, 0).setForeground(QBrush(QColor(96, 64, 31)))
                    self.ui.predictionTableWidget.item(idx, 0).setText(str(round(mean, 6)))
                    # self.ui.predictionTableWidget.item(idx, 0).setToolTip("Suggested value")
        except Exception as e:
            print(e)

    def import_docker(self):
        self.ui.importDockWidget = QtWidgets.QDockWidget(self)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                           QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ui.importDockWidget.sizePolicy().hasHeightForWidth())
        self.ui.importDockWidget.setSizePolicy(sizePolicy)
        self.ui.importDockWidget.setMinimumSize(QtCore.QSize(316, 86))
        self.ui.importDockWidget.setBaseSize(QtCore.QSize(0, 0))
        self.ui.importDockWidget.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.ui.importDockWidget.setObjectName("importDockWidget")
        self.ui.importDockWidgetContents = QtWidgets.QWidget()
        self.ui.importDockWidgetContents.setObjectName("importDockWidgetContents")
        self.ui.importSplitter = QtWidgets.QSplitter(self.ui.importDockWidgetContents)
        self.ui.importSplitter.setGeometry(QtCore.QRect(10, 10, 161, 51))
        self.ui.importSplitter.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.ui.importSplitter.setObjectName("importSplitter")
        self.ui.importLabel = QtWidgets.QLabel(self.ui.importSplitter)
        self.ui.importLabel.setObjectName("importLabel")
        self.ui.importPushButton = QtWidgets.QPushButton(self.ui.importSplitter)
        self.ui.importPushButton.setObjectName("importPushButton")
        self.ui.importDockWidget.setWidget(self.ui.importDockWidgetContents)
        self.ui.importLabel.setText("Import File")
        self.ui.importPushButton.setText("Choose File")
        self.ui.importPushButton.clicked.connect(self.find_csv)
        self.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.ui.importDockWidget)

    def prediction_docker(self):
        self.ui.predictionDockWidget = QtWidgets.QDockWidget(self)
        self.ui.predictionDockWidget.setMinimumSize(QtCore.QSize(316, 330))
        self.ui.predictionDockWidget.setObjectName("predictionDockWidget")
        self.ui.predictionDockWidgetContents = QtWidgets.QWidget()
        self.ui.predictionDockWidgetContents.setObjectName("predictionDockWidgetContents")
        self.ui.layoutWidget = QtWidgets.QWidget(self.ui.predictionDockWidgetContents)
        self.ui.layoutWidget.setGeometry(QtCore.QRect(10, 10, 291, 291))
        self.ui.layoutWidget.setObjectName("layoutWidget")
        self.ui.gridLayout = QtWidgets.QGridLayout(self.ui.layoutWidget)
        self.ui.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.ui.gridLayout.setObjectName("gridLayout")
        self.ui.autofillPushButton = QtWidgets.QPushButton(self.ui.layoutWidget)
        self.ui.autofillPushButton.setObjectName("autofillPushButton")
        self.ui.gridLayout.addWidget(self.ui.autofillPushButton, 2, 0, 1, 1)
        self.ui.predictPushButton = QtWidgets.QPushButton(self.ui.layoutWidget)
        self.ui.predictPushButton.setObjectName("predictPushButton")
        self.ui.gridLayout.addWidget(self.ui.predictPushButton, 2, 1, 1, 1)
        self.ui.predictionLabel = QtWidgets.QLabel(self.ui.layoutWidget)
        self.ui.predictionLabel.setObjectName("predictionLabel")
        self.ui.gridLayout.addWidget(self.ui.predictionLabel, 0, 0, 1, 1)
        self.ui.predictionTableWidget = QtWidgets.QTableWidget(self.ui.layoutWidget)
        self.ui.predictionTableWidget.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.DefaultContextMenu)
        self.ui.predictionTableWidget.setAutoFillBackground(True)
        self.ui.predictionTableWidget.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhPreferNumbers)
        self.ui.predictionTableWidget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.ui.predictionTableWidget.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.ui.predictionTableWidget.setAutoScroll(True)
        self.ui.predictionTableWidget.setProperty("showDropIndicator", True)
        self.ui.predictionTableWidget.setDragDropOverwriteMode(True)
        self.ui.predictionTableWidget.setAlternatingRowColors(True)
        self.ui.predictionTableWidget.setGridStyle(QtCore.Qt.PenStyle.SolidLine)
        self.ui.predictionTableWidget.setObjectName("predictionTableWidget")
        self.ui.predictionTableWidget.setColumnCount(1)
        self.ui.predictionTableWidget.setRowCount(10)
        item = QtWidgets.QTableWidgetItem()
        self.ui.predictionTableWidget.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.ui.predictionTableWidget.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.ui.predictionTableWidget.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.ui.predictionTableWidget.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.ui.predictionTableWidget.setVerticalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.ui.predictionTableWidget.setVerticalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.ui.predictionTableWidget.setVerticalHeaderItem(6, item)
        item = QtWidgets.QTableWidgetItem()
        self.ui.predictionTableWidget.setVerticalHeaderItem(7, item)
        item = QtWidgets.QTableWidgetItem()
        self.ui.predictionTableWidget.setVerticalHeaderItem(8, item)
        item = QtWidgets.QTableWidgetItem()
        self.ui.predictionTableWidget.setVerticalHeaderItem(9, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignTrailing | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.ui.predictionTableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignTrailing | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.ui.predictionTableWidget.setItem(0, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignTrailing | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.ui.predictionTableWidget.setItem(1, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignTrailing | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.ui.predictionTableWidget.setItem(2, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignTrailing | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.ui.predictionTableWidget.setItem(3, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignTrailing | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.ui.predictionTableWidget.setItem(4, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignTrailing | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.ui.predictionTableWidget.setItem(5, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignTrailing | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.ui.predictionTableWidget.setItem(6, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignTrailing | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.ui.predictionTableWidget.setItem(7, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignTrailing | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.ui.predictionTableWidget.setItem(8, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignTrailing | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.ui.predictionTableWidget.setItem(9, 0, item)
        self.ui.predictionTableWidget.horizontalHeader().setVisible(True)
        self.ui.predictionTableWidget.horizontalHeader().setCascadingSectionResizes(False)
        self.ui.predictionTableWidget.horizontalHeader().setSortIndicatorShown(False)
        self.ui.predictionTableWidget.horizontalHeader().setStretchLastSection(True)
        self.ui.predictionTableWidget.verticalHeader().setVisible(True)
        self.ui.predictionTableWidget.verticalHeader().setCascadingSectionResizes(False)
        self.ui.predictionTableWidget.verticalHeader().setHighlightSections(True)
        self.ui.predictionTableWidget.verticalHeader().setSortIndicatorShown(False)
        self.ui.predictionTableWidget.verticalHeader().setStretchLastSection(True)
        self.ui.gridLayout.addWidget(self.ui.predictionTableWidget, 1, 0, 1, 2)
        self.ui.predictionDockWidget.setWidget(self.ui.predictionDockWidgetContents)
        self.ui.autofillPushButton.setText("Autofill")
        self.ui.predictPushButton.setText("Predict")
        self.ui.predictionLabel.setText("Prediction")
        self.ui.predictionTableWidget.setSortingEnabled(False)
        item = self.ui.predictionTableWidget.verticalHeaderItem(0)
        item.setText("WQI 1 year")
        item = self.ui.predictionTableWidget.verticalHeaderItem(1)
        item.setText("WQI 2 years")
        item = self.ui.predictionTableWidget.verticalHeaderItem(2)
        item.setText("WQI 3 years")
        item = self.ui.predictionTableWidget.verticalHeaderItem(3)
        item.setText("Dissolved Oxygen (mg/l)")
        item = self.ui.predictionTableWidget.verticalHeaderItem(4)
        item.setText("pH")
        item = self.ui.predictionTableWidget.verticalHeaderItem(5)
        item.setText("Conductivity (µmhos/cm)")
        item = self.ui.predictionTableWidget.verticalHeaderItem(6)
        item.setText("B.O.D. (mg/l)")
        item = self.ui.predictionTableWidget.verticalHeaderItem(7)
        item.setText("Nitrate (mg/l)")
        item = self.ui.predictionTableWidget.verticalHeaderItem(8)
        item.setText("Fecal Coliform (MPN/100ml)")
        item = self.ui.predictionTableWidget.verticalHeaderItem(9)
        item.setText("Total Coliform (MPN/100ml)")
        item = self.ui.predictionTableWidget.horizontalHeaderItem(0)
        item.setText("Value")
        __sortingEnabled = self.ui.predictionTableWidget.isSortingEnabled()
        self.ui.predictionTableWidget.setSortingEnabled(False)
        self.ui.predictionTableWidget.setSortingEnabled(__sortingEnabled)
        self.ui.predictionTableWidget.setItemDelegateForColumn(0, self.delegate)
        # self.ui.predictPushButton.setEnabled(False)
        self.ui.predictPushButton.clicked.connect(self.prediction)
        # self.ui.autofillPushButton.setEnabled(False)
        self.ui.autofillPushButton.clicked.connect(self.auto_fill)
        self.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.ui.predictionDockWidget)

    def result_docker(self):
        self.ui.resultDockWidget = QtWidgets.QDockWidget(self)
        self.ui.resultDockWidget.setMinimumSize(QtCore.QSize(316, 170))
        self.ui.resultDockWidget.setObjectName("resultDockWidget")
        self.ui.resultDockWidgetContents = QtWidgets.QWidget()
        self.ui.resultDockWidgetContents.setObjectName("resultDockWidgetContents")
        self.ui.resultSplitter = QtWidgets.QSplitter(self.ui.resultDockWidgetContents)
        self.ui.resultSplitter.setGeometry(QtCore.QRect(10, 10, 291, 141))
        self.ui.resultSplitter.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.ui.resultSplitter.setObjectName("resultSplitter")
        self.ui.resultLabel = QtWidgets.QLabel(self.ui.resultSplitter)
        self.ui.resultLabel.setObjectName("resultLabel")
        self.ui.resultOutput = QtWidgets.QTextEdit(self.ui.resultSplitter)
        self.ui.resultOutput.setObjectName("resultOutput")
        self.ui.resultDockWidget.setWidget(self.ui.resultDockWidgetContents)
        self.ui.resultLabel.setText("Result")
        self.ui.resultOutput.setHtml(
            "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
            "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
            "p, li { white-space: pre-wrap; }\n"
            "</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400;"
            " font-style:normal;\">\n"
            "<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px;"
            "-qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>")
        self.ui.resultOutput.setReadOnly(True)
        self.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.ui.resultDockWidget)

    def build_about(self):
        dialog = QDialog(self)
        dialog.ui = AboutDialog()
        dialog.ui.setupUi(dialog)
        dialog.exec()
        dialog.show()


class MyToolBar(NavigationToolbar):

    def zoom(self, *args):
        super().zoom(*args)
        self._update_buttons_checked()
        self.locLabel.setText("")

    def pan(self, *args):
        super().pan(*args)
        self._update_buttons_checked()
        self.locLabel.setText("")

    def set_message(self, s):
        pass
        # self.message.emit(s)
        if self.coordinates:
            self.locLabel.setText(s)


class TableWidgetDelegate(QItemDelegate):
    def createEditor(self, parent: QWidget, option, index: QtCore.QModelIndex) -> QWidget:
        editor = QLineEdit(parent=parent)
        pattern = '|[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'
        reg = QRegularExpression(pattern)
        reg_validator = QRegularExpressionValidator(reg)
        editor.setValidator(reg_validator)
        return editor


if __name__ == "__main__":
    app = Qtw.QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec())
