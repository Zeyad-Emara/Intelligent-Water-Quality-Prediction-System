# importing Qt widgets
from PyQt6.QtWidgets import *
import sys

# importing pyqtgraph as pg
import pyqtgraph as pg



class Window(QMainWindow):

	def __init__(self):
		super().__init__()

		# setting title
		self.setWindowTitle("PyQtGraph")

		# setting geometry
		self.setGeometry(100, 100, 600, 500)

		# calling method
		self.UiComponents()

		# showing all the widgets
		self.show()

	# method for components
	def UiComponents(self):

		# creating a widget object
		widget = QWidget()

		# creating a push button object
		btn = QPushButton('Push Button')

		# creating a line edit widget
		text = QLineEdit("Line Edit")

		# creating a check box widget
		check = QCheckBox("Check Box")

		# creating a plot window
		plot = pg.plot()

		# create list for y-axis
		y1 = [5, 5, 7, 10, 3, 8, 9, 1, 6, 2]

		# create horizontal list i.e x-axis
		x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

		# create pyqt5graph bar graph item
		# with width = 0.6
		# with bar colors = green
		bargraph = pg.BarGraphItem(x = x, height = y1, width = 0.6, brush ='g')

		# add item to plot window
		# adding bargraph item to the plot window
		plot.addItem(bargraph)

		# Creating a grid layout
		layout = QGridLayout()

		# setting this layout to the widget
		widget.setLayout(layout)

		# adding widgets in the layout in their proper positions
		# button goes in upper-left
		layout.addWidget(btn, 0, 0)

		# text edit goes in middle-left
		layout.addWidget(text, 1, 0)

		# check box widget goes in bottom-left
		layout.addWidget(check, 3, 0)

		# plot window goes on right side, spanning 3 rows
		layout.addWidget(plot, 0, 1, 3, 1)

		# setting this widget as central widget of the main window
		self.setCentralWidget(widget)


if __name__ == "__main__":
	# create pyqt5 app
	App = QApplication(sys.argv)

	# create the instance of our Window
	window = Window()

	# start the app
	sys.exit(App.exec())
