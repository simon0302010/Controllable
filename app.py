import sys

try:
    import cv2

    if hasattr(cv2, 'qt'):
        print("ERROR: opencv-python or opencv-contrib-python is installed!")
        print("This conflicts with PyQt5. Please run:")
        print("  pip uninstall opencv-python opencv-contrib-python")
        print("  pip install opencv-python-headless")
        sys.exit(1)
except ImportError:
    pass

from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QSizePolicy, QPushButton
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import pyqtSlot, Qt
import numpy as np
import video_feed


class App(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Controllable")
        self.display_width = 640
        self.display_height = 480

        self.header_text = QLabel("Controllable")
        self.header_text.setFont(QFont("", 20))
        self.header_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.header_text.setAlignment(Qt.AlignCenter)

        self.info_text = QLabel("""Please put one hand in front of your camera so that it's fully visible and in a
        comfortable distance from it. Press "Calibrate" when you're ready.""")
        self.info_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.info_text.setAlignment(Qt.AlignCenter)

        self.push_button = QPushButton("Calibrate")
        self.push_button.setFixedWidth(100)
        self.push_button.clicked.connect(self.on_calibrate)

        self.image_label = QLabel(self)
        self.image_label.resize(self.display_width, self.display_height)

        # placeholder for video feed
        pixmap = QPixmap(self.display_width, self.display_height)
        pixmap.fill(Qt.white)
        painter = QtGui.QPainter(pixmap)
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "Loading video feed...")
        painter.end()
        self.image_label.setPixmap(pixmap)

        vbox = QVBoxLayout()
        vbox.addWidget(self.header_text)
        vbox.addWidget(self.info_text)
        vbox.addWidget(self.push_button, alignment=Qt.AlignCenter)
        vbox.addWidget(self.image_label)
        self.setLayout(vbox)

        self.video_thread = video_feed.VideoThread()
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.start()

    def closeEvent(self, event) -> None:
        self.video_thread.stop()
        event.accept()

    def on_calibrate(self):
        self.video_thread.trigger_calibration.emit()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
