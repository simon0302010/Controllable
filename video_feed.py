from PyQt5.QtCore import QObject, pyqtSignal, QThread
import cv2
import sys
import math
import mediapipe as mp
from screeninfo import get_monitors
from mouse_interpolator import MouseInterpolator
import time
from pynput.mouse import Button, Controller
import numpy as np
import landmarkers

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    
    def __init__(self) -> None:
        super().__init__()
        self._run_flag = True

    def map_coordinate(self, coord, old_min=0.2, old_max=0.8):
        coord = max(old_min, min(old_max, coord))
        return (coord - old_min) / (old_max - old_min)

    def run(self):
        cap = cv2.VideoCapture(0)
        click_distance = self.calibrate(cap)
        monitor = get_monitors()[0]

        hand_landmarker = landmarkers.Landmarker()
        mouse_interpolator = MouseInterpolator()
        pynput_mouse = Controller()
        last_click = time.time()
        click_since = None
        action = None
        
        x_array = []
        y_array = []

        last_touch = time.time()
        
        last_hand_detected = time.time()
        is_dragging = False
        
        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            try:
                height, width, _ = frame.shape
            except AttributeError:
                print("No camera detected.")
                sys.exit(1)
            monitor_width, monitor_height = monitor.width, monitor.height
            hand_landmarker.detect_async(frame)
            if hand_landmarker.result and hand_landmarker.result.hand_landmarks and len(hand_landmarker.result.hand_landmarks[0]) == 21:
                last_hand_detected = time.time()
                pointer_tip = hand_landmarker.result.hand_landmarks[0][8]
                thumb_tip = hand_landmarker.result.hand_landmarks[0][4]
                                
                distance = math.dist(
                    [thumb_tip.x, thumb_tip.y],
                    [pointer_tip.x, pointer_tip.y]
                )
                
                if distance <= click_distance and not click_since:
                    click_since = time.time()

                if distance <= click_distance:
                    last_touch = time.time()
                    if click_since and time.time() - click_since >= 0.2:
                        action = "drag"
                        is_dragging = True
                    else:
                        action = "hold"
                elif click_since:
                    if time.time() - click_since < 0.2:
                        action = "click"
                    else:
                        action = None
                    click_since = None
                    is_dragging = False
                else:
                    action = None

                print(action)
                
                if time.time() - last_touch < 0.1:
                    distance = click_distance
                should_click = distance <= click_distance and (time.time() - last_click) > 0.5
                
                if distance <= click_distance:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                
                norm_x, norm_y = pointer_tip.x, pointer_tip.y
                mapped_x = self.map_coordinate(pointer_tip.x)
                mapped_y = self.map_coordinate(pointer_tip.y)
                pixel_x = int(norm_x * width)
                pixel_y = int(norm_y * height)
                monitor_x = int(mapped_x * monitor_width)
                monitor_y = int(mapped_y * monitor_height)
                x_array.append(monitor_x)
                y_array.append(monitor_y)
                cv2.circle(frame, (pixel_x, pixel_y), 5, color, -1)
                
                thumb_pixel_x = int(thumb_tip.x * width)
                thumb_pixel_y = int(thumb_tip.y * height)
                cv2.circle(frame, (thumb_pixel_x, thumb_pixel_y), 5, color, -1)
                
                cv2.line(frame, (pixel_x, pixel_y), (thumb_pixel_x, thumb_pixel_y), color, 2)
                
                if len(x_array) >= 5:
                    recent_x = x_array[-min(10, len(x_array)):]
                    recent_y = y_array[-min(10, len(y_array)):]
                    
                    # the lower the smoother
                    alpha = 0.3
                    smoothed_x = recent_x[0]
                    smoothed_y = recent_y[0]
                    for i in range(1, len(recent_x)):
                        smoothed_x = alpha * recent_x[i] + (1 - alpha) * smoothed_x
                        smoothed_y = alpha * recent_y[i] + (1 - alpha) * smoothed_y
                    
                    monitor_x = int(smoothed_x)
                    monitor_y = int(smoothed_y)
                    
                mouse_interpolator.move_to(monitor_x, monitor_y)
                try:
                    if action == "click" and should_click:
                        pynput_mouse.click(Button.left, 1)
                        last_click = time.time()
                    elif is_dragging or action == "drag":
                        pynput_mouse.press(Button.left)
                    else:
                        pynput_mouse.release(Button.left)
                except RuntimeError:
                    pass
                
            else:
                if time.time() - last_hand_detected > 0.3:
                    is_dragging = False
                    action = None
            
            self.change_pixmap_signal.emit(frame)
        
        hand_landmarker.close()
        cap.release()
        
    def stop(self):
        self._run_flag = False
        self.wait()
        
    def calibrate(self, cap):
        hand_landmarker = landmarkers.Landmarker()

        calibration_time = 0.0
        last_hand_time = time.time()

        touching = []
        not_touching = []
        click_distance = 0.0
        
        ret, frame = cap.read()
        try:
            height, width, _ = frame.shape
        except AttributeError:
            print("No camera detected.")
            sys.exit(1)
        
        text = None
        font_scale = width / 640.0  # Scale based on camera resolution
        thickness = max(2, int(font_scale * 2))
        
        # Calculate text position dynamically
        text_x = int(width * 0.02)
        text_y = int(height * 0.9)
        
        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            
            try:
                height, width, _ = frame.shape
            except AttributeError:
                print("No camera detected.")
                sys.exit(1)
            
            hand_landmarker.detect_async(frame)
            
            try:
                if hand_landmarker.result and hand_landmarker.result.hand_landmarks and len(hand_landmarker.result.hand_landmarks[0]) == 21:
                    pointer_tip = hand_landmarker.result.hand_landmarks[0][8]
                    thumb_tip = hand_landmarker.result.hand_landmarks[0][4]
                                
                    distance = math.dist(
                        [thumb_tip.x, thumb_tip.y],
                        [pointer_tip.x, pointer_tip.y]
                    )
                    
                    now = time.time()
                    calibration_time += now - last_hand_time
                    last_hand_time = now
                    
                    elapsed_time = calibration_time
                    
                    if elapsed_time <= 5:
                        countdown = int(6 - elapsed_time)
                        text = f"Bring your thumb tip and pointer tip close (but not touching) in {countdown} seconds"
                        print(f"\r{text}", end='', flush=True)
                        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), 4, cv2.LINE_AA)
                    elif elapsed_time <= 8:
                        countdown = int(9 - elapsed_time)
                        text = f"Measuring not touching position... {countdown} seconds remaining"
                        print(f"\r{text}", end='', flush=True)
                        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), 4, cv2.LINE_AA)
                        not_touching.append(distance)
                    elif elapsed_time <= 11:
                        countdown = int(12 - elapsed_time)
                        text = f"Touch your thumb tip and pointer tip together in {countdown} seconds"
                        print(f"\r{text}", end='', flush=True)
                        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), 4, cv2.LINE_AA)
                    elif elapsed_time <= 14:
                        countdown = int(15 - elapsed_time)
                        text = f"Measuring touching position... {countdown} seconds remaining"
                        print(f"\r{text}", end='', flush=True)
                        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), 4, cv2.LINE_AA)
                        touching.append(distance)
                    else:
                        # Calibration complete
                        touching_threshold = max(touching) if touching else 0.04
                        not_touching_threshold = min(not_touching) if not_touching else 0.1
                        click_distance = (touching_threshold + not_touching_threshold) / 2
                        click_distance = max(0.04, min(0.15, click_distance))
                        print(f"\n\nCalibration complete! Click Distance: {click_distance:.4f}")
                        print(f"Touching range: {touching_threshold:.4f}, Not touching range: {not_touching_threshold:.4f}")
                        
                        hand_landmarker.close()
                        return click_distance
                else:
                    if text:
                        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), 4, cv2.LINE_AA)
                
            except (AttributeError, IndexError):
                last_hand_time = time.time()
                
            self.change_pixmap_signal.emit(frame)
    
        hand_landmarker.close()
        return click_distance