import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtCore import QTimer
import pyqtgraph as pg
import numpy as np

class RobotWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Robot Visualization')
        self.setGeometry(100, 100, 800, 600)

        # Create widgets
        self.robot_label = QLabel('Robot Visualization')
        self.position_plot_label = QLabel('Position Plot')
        self.position_plot_widget = pg.PlotWidget()
        self.position_plot_widget.setBackground('w')  # Set background color to white
        self.position_plot_curve = self.position_plot_widget.plot(pen='b')  # Plot with a blue pen

        # Buttons
        self.start_button = QPushButton('Start')
        self.stop_button = QPushButton('Stop')
        self.plot_button = QPushButton('Plot Position')

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.robot_label)
        layout.addWidget(self.position_plot_label)
        layout.addWidget(self.position_plot_widget)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.plot_button)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Connect button signals to slots
        self.start_button.clicked.connect(self.startRobot)
        self.stop_button.clicked.connect(self.stopRobot)
        self.plot_button.clicked.connect(self.plotPosition)

        # Create QTimer for continuous updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateRobotPosition)

        # Initialize data for live plotting
        self.time_points = []
        self.position_values = []

    def startRobot(self):
        # Start the QTimer for continuous updates
        self.timer.start(100)  # Update every 100 milliseconds
        self.robot_label.setText('Robot Started')

    def stopRobot(self):
        # Stop the QTimer
        self.timer.stop()
        self.robot_label.setText('Robot Stopped')

    def plotPosition(self):

        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        self.position_plot_curve.setData(x, y)

    def updateRobotPosition(self):
        # Example: Update robot position
        # Add your logic to update the robot position
        # For demonstration purposes, a random position is plotted
        time_point = len(self.time_points)
        position_value = np.random.uniform(-1, 1)
        
        self.time_points.append(time_point)
        self.position_values.append(position_value)

        # Update the live plot
        self.position_plot_curve.setData(self.time_points, self.position_values)

        self.robot_label.setText(f'Robot Position: ({time_point}, {position_value:.2f})')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RobotWindow()
    window.show()
    sys.exit(app.exec_())
