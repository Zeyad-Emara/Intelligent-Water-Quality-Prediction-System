import sys

from PyQt6.QtWidgets import QLineEdit, QItemDelegate, QWidget, QComboBox, QTableView, QStyledItemDelegate, QDialog

from template import Ui_MainWindow
from about_template import Ui_Dialog as AboutDialog
from PyQt6.QtCore import QRegularExpression, QRect
from PyQt6 import QtWidgets as Qtw, QtCore, QtGui
from PyQt6.QtGui import QDoubleValidator, QValidator, QRegularExpressionValidator, QPen, QColor

# for exe file compilation
import sklearn.utils._typedefs


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
        delegate = TableWidgetDelegate()
        self.ui.predictionTableWidget.setItemDelegateForColumn(0, delegate)

        # add default model
        try:
            self.reg = load('resource/WQIModelv1.pkl')
            self.std = load('resource/StdScaler.pkl')
        except Exception:
            self.ui.statusbar.showMessage("No modal detected")

        # add about page
        self.ui.actionAbout.triggered.connect(self.build_about)

        # Your code ends here
        self.show()

    def build_about(self):
        dialog = QDialog(self)
        dialog.ui = AboutDialog()
        dialog.ui.setupUi(dialog)
        dialog.exec()
        dialog.show()

    def find_csv(self):
        try:
            self.filePath = Qtw.QFileDialog.getOpenFileName(filter="csv (*.csv)")[0]
        except Exception:
            self.ui.statusbar.showMessage("No dataset selected")
        self.ui.statusbar.showMessage("")
        self.plot_graph()

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
        value = self.read_table_data()
        if value:
            value = np.array(value).reshape(1, -1)
            self.predict_input = self.std.transform(value)
            self.predict_value = self.reg.predict(self.predict_input)
            self.ui.resultOutput.clear()
            self.ui.resultOutput.insertPlainText(str(self.predict_value[0]))

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
