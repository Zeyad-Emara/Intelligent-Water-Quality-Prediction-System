import sys
from template import Ui_MainWindow
from PyQt6 import QtCore as qtc
from PyQt6 import QtWidgets as qtw

import numpy as np

from matplotlib.backends.backend_qtagg import (FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure


class Window(qtw.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(Window, self).__init__(*args, **kwargs)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.resize(1000, 650)
        # Your code starts here

        # Plot graph and Toolbar
        layout = qtw.QVBoxLayout()
        static_canvas = FigureCanvasQTAgg(Figure(figsize=(10, 10)))
        layout.addWidget(NavigationToolbar(static_canvas, self))
        graph = static_canvas.figure.subplots()
        t = np.linspace(0, 10, 501)
        graph.plot(t, np.tan(t), ".")
        layout.addWidget(static_canvas)
        self.ui.graphWidget.setLayout(layout)

        # Your code ends here
        self.show()


if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec())
