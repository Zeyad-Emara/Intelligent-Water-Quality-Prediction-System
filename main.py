import sys
from template import Ui_MainWindow
from PyQt6 import QtCore as qtc
from PyQt6 import QtWidgets as qtw

import numpy as np

from matplotlib.backends.backend_qtagg import (FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


class Window(qtw.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(Window, self).__init__(*args, **kwargs)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.resize(1000, 650)

        # Your code starts here
        self.filePath = None
        self.predict_input = None
        self.predict_value = None
        self.reg = None
        self.data_frame = None

        # Update statusbar
        self.ui.statusbar.showMessage("No dataset found")

        # Draw canvas and Toolbar
        self.layout = qtw.QVBoxLayout()
        self.static_canvas = FigureCanvasQTAgg(Figure(figsize=(10, 10)))
        self.layout.addWidget(NavigationToolbar(self.static_canvas, self))
        self.graph = self.static_canvas.figure.subplots()
        self.layout.addWidget(self.static_canvas)
        self.ui.graphWidget.setLayout(self.layout)

        # add import button function
        self.ui.importPushButton.clicked.connect(self.find_csv)

        # add predicting value function
        self.ui.predictPushButton.clicked.connect(self.prediction)

        # set result display area read only
        self.ui.resultOutput.setReadOnly(True)

        # Your code ends here
        self.show()

    def find_csv(self):
        self.filePath = qtw.QFileDialog.getOpenFileName(filter="csv (*.csv)")[0]
        self.build_model()

    def build_model(self):
        df = pd.read_csv(self.filePath, encoding='utf-8')
        df_dropna = df.dropna()
        X = df_dropna.iloc[:, 0:-1]
        y = df_dropna.iloc[:, -1]
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=89)
        self.reg = SVC()
        self.reg.fit(X_train, y_train)
        self.data_frame = df
        self.plot_graph()
        self.statusBar().showMessage("")

    def plot_graph(self):
        columns = list(self.data_frame.columns)
        df = pd.read_csv(self.filePath, usecols=columns)
        df.plot(ax=self.graph)
        self.static_canvas.draw()

    def prediction(self):
        try:
            value = self.read_table_data()
            value = np.array(value).reshape(1, -1)
            self.predict_input = StandardScaler().fit_transform(value)
            self.predict_value = self.reg.predict(value)
            self.ui.resultOutput.clear()
            self.ui.resultOutput.insertPlainText(str(self.predict_value[0]))
        except:
            self.ui.statusbar.showMessage("")

    def read_table_data(self):
        item = []
        row_count = self.ui.predictionTableWidget.rowCount()
        for row in range(row_count):
            item.append(float(self.ui.predictionTableWidget.item(row, 0).text()))
        return item


if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec())
