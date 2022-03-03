import sys

from PyQt6.QtWidgets import QLineEdit, QItemDelegate, QWidget, QComboBox, QTableView, QStyledItemDelegate, QDialog

import template
from template import Ui_MainWindow
from about_template import Ui_Dialog as AboutDialog
from PyQt6.QtCore import QRegularExpression, QRect, QCoreApplication
from PyQt6 import QtWidgets as Qtw, QtCore, QtGui, QtWidgets
from PyQt6.QtGui import QDoubleValidator, QValidator, QRegularExpressionValidator, QPen, QColor

# for exe file compilation
import sklearn.utils._typedefs
from PyQt6 import QtCore, QtGui, QtWidgets


from matplotlib.backends.backend_qtagg import (FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from joblib import dump, load


class Window(Qtw.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(Window, self).__init__(*args, **kwargs)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.resize(1000, 650)
        self.std = None

        # Your code starts here
        self.filePath = None
        self.predict_input = None
        self.predict_value = None
        self.reg = None
        self.data_frame = None

        # Update statusbar
        self.ui.statusbar.showMessage("No dataset found")

        # Draw canvas and Toolbar
        self.layout = Qtw.QVBoxLayout()
        self.static_canvas = FigureCanvasQTAgg(Figure(figsize=(10, 10)))
        self.layout.addWidget(MyToolBar(self.static_canvas, self.centralWidget()))
        self.graph = self.static_canvas.figure.add_subplot(111)
        self.layout.addWidget(self.static_canvas)
        self.ui.verticalLayout.addLayout(self.layout)

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

        # add default model
        try:
            self.reg = load('resource/WQIModelv1.pkl')
            self.std = load('resource/StdScaler.pkl')
        except Exception:
            self.ui.statusbar.showMessage("No modal detected")

        # add about page
        self.ui.actionAbout.triggered.connect(self.build_about)

        # add close function
        self.ui.actionClose.triggered.connect(QCoreApplication.instance().quit)

        # add import docker widget
        self.ui.actionImport_widget.triggered.connect(self.import_docker)

        # add prediction docker widget
        self.ui.actionPrediction_widget.triggered.connect(self.prediction_docker)

        # add result docker widget
        self.ui.actionResult_widget.triggered.connect(self.result_docker)

        # Your code ends here
        self.show()

    def find_csv(self):
        try:
            self.filePath = Qtw.QFileDialog.getOpenFileName(filter="csv (*.csv)")[0]
        except Exception:
            self.ui.statusbar.showMessage("No dataset selected")


    def build_model(self): #technically unused for now
        df = pd.read_csv(self.filePath, encoding='utf-8')
        df_dropna = df.dropna()
        X = df_dropna.iloc[:, 0:-1]
        y = df_dropna.iloc[:, -1]
        self.std = StandardScaler()
        X = self.std.fit_transform(X.values)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=89)
        self.reg = SVC()
        self.reg.fit(X_train, y_train)
        self.data_frame = df
        self.ui.statusbar.showMessage("")
        #self.plot_graph()

    def plot_graph(self):
        try:
            self.data_frame = pd.read_csv(self.filePath, encoding='utf-8')
            columns = list(self.data_frame.columns)
            df = pd.read_csv(self.filePath, usecols=columns)
            df.plot(ax=self.graph)
            self.graph.legend().set_draggable(True)
            self.graph.axes.set_xlabel('X Axis')
            self.graph.axes.set_ylabel('Y Axis')
            self.graph.axes.set_title('Title')
            self.static_canvas.draw()
            self.ui.predictPushButton.setEnabled(True)
            self.ui.autofillPushButton.setEnabled(True)
        except Exception:
            self.ui.statusbar.showMessage("Failed to plot graph")

    def prediction(self):

        try:
            self.reg = load('resource/models/WQIModelv1.pkl')
            self.std = load('resource/models/StdScaler.pkl')
            
            value = self.read_table_data()
            value = np.array(value).reshape(1, -1)
            self.predict_input = self.std.transform(value)
            self.predict_value = self.reg.predict(self.predict_input)
            self.ui.resultOutput.clear()
            self.ui.resultOutput.insertPlainText(str(self.predict_value[0]))
        except Exception as e:
            print(e)
            self.ui.statusbar.showMessage("fail")

    def read_table_data(self):
        self.ui.statusbar.showMessage("")
        # dissolved oxygen, pH, conductivity, B.O.D., Nitrate, Fecal Coliform, Total Coliform
        item = []
        row_count = self.ui.predictionTableWidget.rowCount()
        # validator = QDoubleValidator(0,100,14)
        for row in range(row_count):
            # check none input
            if self.ui.predictionTableWidget.item(row, 0) is None:
                self.ui.statusbar.showMessage("Please fill up all the rows")
                break
            value = float(self.ui.predictionTableWidget.item(row, 0).text())
            # check row 0
            if row == 0 and value not in range(-1, 100):
                self.ui.predictionTableWidget.item(row, 0).setBackground(QColor("red"))
                continue
            else:
                self.ui.predictionTableWidget.item(row, 0).setBackground(QColor("white"))
            # check row 1
            if row == 1 and value not in range(-1, 14):
                self.ui.predictionTableWidget.item(row, 0).setBackground(QColor("red"))
                continue
            else:
                self.ui.predictionTableWidget.item(row, 0).setBackground(QColor("white"))
            # check row 2
            if row == 2 and value not in range(-1, 1000):
                self.ui.predictionTableWidget.item(row, 0).setBackground(QColor("red"))
                continue
            else:
                self.ui.predictionTableWidget.item(row, 0).setBackground(QColor("white"))
            # check row 3
            if row == 3 and value not in range(-1, 100):
                self.ui.predictionTableWidget.item(row, 0).setBackground(QColor("red"))
                continue
            else:
                self.ui.predictionTableWidget.item(row, 0).setBackground(QColor("white"))
            # check row 4
            if row == 4 and value not in range(-1, 20):
                self.ui.predictionTableWidget.item(row, 0).setBackground(QColor("red"))
                continue
            else:
                self.ui.predictionTableWidget.item(row, 0).setBackground(QColor("white"))
            # check row 5
            if row == 5 and value not in range(-1, 10000):
                self.ui.predictionTableWidget.item(row, 0).setBackground(QColor("red"))
                continue
            else:
                self.ui.predictionTableWidget.item(row, 0).setBackground(QColor("white"))
            # check row 6
            if row == 6 and value not in range(-1, 100000):
                self.ui.predictionTableWidget.item(row, 0).setBackground(QColor("red"))
                continue
            else:
                self.ui.predictionTableWidget.item(row, 0).setBackground(QColor("white"))
            item.append(value)
        if len(item) == 7:
            return item
        return False

    def auto_fill(self):
        try:
            mean = self.data_frame['PH'].mean()
            # self.ui.predictionTableWidget.item(1, 0).setData(1)
            self.ui.predictionTableWidget.item(1, 0).setText(str(mean))
            self.ui.predictionTableWidget.item(1, 0).setToolTip("Suggested value")
            self.ui.actionAnalysis_data.setStatusTip("hello")
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
        self.ui.predictionDockWidget.setMinimumSize(QtCore.QSize(316, 322))
        self.ui.predictionDockWidget.setObjectName("predictionDockWidget")
        self.ui.predictionDockWidgetContents = QtWidgets.QWidget()
        self.ui.predictionDockWidgetContents.setObjectName("predictionDockWidgetContents")
        self.ui.layoutWidget = QtWidgets.QWidget(self.ui.predictionDockWidgetContents)
        self.ui.layoutWidget.setGeometry(QtCore.QRect(10, 10, 291, 291))
        self.ui.layoutWidget.setObjectName("layoutWidget")
        self.ui.gridLayout = QtWidgets.QGridLayout(self.ui.layoutWidget)
        self.ui.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.ui.gridLayout.setObjectName("gridLayout")
        self.ui.predictionLabel = QtWidgets.QLabel(self.ui.layoutWidget)
        self.ui.predictionLabel.setObjectName("predictionLabel")
        self.ui.gridLayout.addWidget(self.ui.predictionLabel, 0, 0, 1, 1)
        self.ui.autofillPushButton = QtWidgets.QPushButton(self.ui.layoutWidget)
        self.ui.autofillPushButton.setObjectName("autofillPushButton")
        self.ui.gridLayout.addWidget(self.ui.autofillPushButton, 2, 0, 1, 1)
        self.ui.predictPushButton = QtWidgets.QPushButton(self.ui.layoutWidget)
        self.ui.predictPushButton.setObjectName("predictPushButton")
        self.ui.gridLayout.addWidget(self.ui.predictPushButton, 2, 1, 1, 1)
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
        self.ui.predictionTableWidget.setRowCount(7)
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
        self.ui.predictionTableWidget.setHorizontalHeaderItem(0, item)
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
        self.ui.predictPushButton.clicked.connect(self.prediction)
        self.ui.predictionLabel.setText("Prediction")
        self.ui.autofillPushButton.setText("Autofill")
        self.ui.predictPushButton.setText("Predict")
        self.ui.predictionTableWidget.setSortingEnabled(False)
        item = self.ui.predictionTableWidget.verticalHeaderItem(0)
        item.setText("Dissolved Oxygen (mg/l)")
        item = self.ui.predictionTableWidget.verticalHeaderItem(1)
        item.setText("pH")
        item = self.ui.predictionTableWidget.verticalHeaderItem(2)
        item.setText("Conductivity (µmhos/cm)")
        item = self.ui.predictionTableWidget.verticalHeaderItem(3)
        item.setText("B.O.D. (mg/l)")
        item = self.ui.predictionTableWidget.verticalHeaderItem(4)
        item.setText("Nitrate (mg/l)")
        item = self.ui.predictionTableWidget.verticalHeaderItem(5)
        item.setText("Fecal Coliform (MPN/100ml)")
        item = self.ui.predictionTableWidget.verticalHeaderItem(6)
        item.setText("Total Coliform (MPN/100ml)")
        item = self.ui.predictionTableWidget.horizontalHeaderItem(0)
        item.setText("Value")
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
        self.ui.resultOutput.setHtml("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                     "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                     "p, li { white-space: pre-wrap; }\n"
                                     "</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
                                     "<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>")
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
        # if self.coordinates:
        #     self.locLabel.setText(s)


class TableWidgetDelegate(QItemDelegate):
    def createEditor(self, parent: QWidget, option, index: QtCore.QModelIndex) -> QWidget:
        editor = QLineEdit(parent=parent)
        pattern = '[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'
        reg = QRegularExpression(pattern)
        reg_validator = QRegularExpressionValidator(reg)
        editor.setValidator(reg_validator)
        return editor


# class TestDelegate(QStyledItemDelegate):
#     def __init__(self):
#         super().__init__()
#
#     def paint(self, painter: QtGui.QPainter, option: 'QStyleOptionViewItem', index: QtCore.QModelIndex) -> None:
#         super().paint(painter, option, index)
#
#         painter.save()
#         pen = QPen(QColor("red"))
#         qr = QRect(option.rect)
#         qr.setWidth(pen.width())
#         painter.setPen(pen)
#         painter.drawRect(qr)
#         painter.restore()


if __name__ == "__main__":
    app = Qtw.QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec())
