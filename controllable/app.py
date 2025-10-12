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
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QSizePolicy, QPushButton, QCheckBox
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import pyqtSlot, Qt
from tqdm import tqdm
import numpy as np
import requests
import sys
import os

from . import video_feed


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

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

        # settings
        settings_header = QLabel("Settings")
        settings_header.setFont(QFont("", 14))
        settings_header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        settings_header.setAlignment(Qt.AlignCenter)

        self.enable_dragging_box = QCheckBox("Enable support for dragging")
        self.enable_dragging_box.setToolTip("Enable support for dragging (More prone to accidental inputs)")
        self.enable_dragging_box.stateChanged.connect(self.on_dragging_changed)

        self.only_hand = QCheckBox("Only show hand landmarkers")
        self.only_hand.setToolTip("Only shows the hand landmarkers in the video feed. Useful for demos.")
        self.only_hand.stateChanged.connect(self.on_only_hand_changed)

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
        vbox.addWidget(settings_header)
        vbox.addWidget(self.enable_dragging_box, alignment=Qt.AlignCenter)
        vbox.addWidget(self.only_hand, alignment=Qt.AlignCenter)
        self.setLayout(vbox)

        self.video_thread = video_feed.VideoThread()
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.change_text_signal.connect(self.change_text)
        self.video_thread.calibration_completed_signal.connect(self.calibration_completed)
        self.video_thread.start()

        if "--only-hand" in sys.argv:
            self.only_hand.setChecked(True)
            self.video_thread.only_hand = True

        self.calibrated = False

        # download model
        if not os.path.exists("hand_landmarker.task"):
            self.download_model()

    def closeEvent(self, event) -> None:
        self.video_thread.stop()
        event.accept()

    def on_calibrate(self):
        self.push_button.setDisabled(True)
        self.video_thread.trigger_calibration.emit()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    @pyqtSlot(str)
    def change_text(self, new_text):
        self.info_text.setText(new_text)

    @pyqtSlot()
    def calibration_completed(self):
        self.calibrated = True
        self.push_button.setText("Start")
        self.push_button.setDisabled(False)
        self.push_button.clicked.disconnect()
        self.push_button.clicked.connect(self.begin)

    def begin(self):
        self.video_thread.began_processing = True
        self.info_text.setText("Running. Tap your index finger and thumb to click. Dragging can be enabled in the settings.")
        self.push_button.setText("Stop")
        self.push_button.clicked.disconnect()
        self.push_button.clicked.connect(self.stop)

    def stop(self):
        self.video_thread.began_processing = False
        self.info_text.setText("Stopped. Press the start button to start processing again.")
        self.push_button.setText("Start")
        self.push_button.clicked.disconnect()
        self.push_button.clicked.connect(self.begin)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def on_dragging_changed(self, state):
        self.video_thread.enable_dragging = (state == Qt.Checked)

    def on_only_hand_changed(self, state):
        self.video_thread.only_hand = (state == Qt.Checked)

    def download_model(self):
        fname = MODEL_URL.split("/")[-1]
        resp = requests.get(MODEL_URL, stream=True)
        total = int(resp.headers.get('content-length', 0))
        with open(fname, 'wb') as file, tqdm(
                desc=fname,
                total=total,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for data in resp.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)

def main():
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()