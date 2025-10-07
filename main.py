import cv2
import sys
import math
import mediapipe as mp
import pyautogui
from mediapipe.framework.formats import landmark_pb2
from screeninfo import get_monitors
from mouse_interpolator import MouseInterpolator
import time
from pynput.mouse import Button, Controller


class Landmarker():
    def __init__(self) -> None:
        self.result = None
        self.landmarker = mp.tasks.vision.HandLandmarker
        self.timestamp = 0
        self.createLandmarker()
        
    def createLandmarker(self):
        def update_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            self.result = result
        
        base_options = mp.tasks.BaseOptions(
            model_asset_path="hand_landmarker.task",
             delegate=mp.tasks.BaseOptions.Delegate.GPU
        )
        
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options = base_options,
            running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands = 2,
            min_hand_detection_confidence = 0.3,
            min_hand_presence_confidence = 0.3,
            min_tracking_confidence = 0.3,
            result_callback=update_result
        )
        
        self.landmarker = self.landmarker.create_from_options(options)
        
    def detect_async(self, frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.timestamp += 1
        self.landmarker.detect_async(image = mp_image, timestamp_ms=self.timestamp)
        
    def close(self):
        self.landmarker.close()

def map_coordinate(coord, old_min=0.2, old_max=0.8):
    coord = max(old_min, min(old_max, coord))
    return (coord - old_min) / (old_max - old_min)

def main():
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    monitor = get_monitors()[0]

    hand_landmarker = Landmarker()
    mouse_interpolator = MouseInterpolator()
    pynput_mouse = Controller()
    last_click = time.time()
    
    x_array = []
    y_array = []

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        try:
            height, width, _ = frame.shape
        except AttributeError:
            print("No camera detected.")
            sys.exit(1)
            
        monitor_width, monitor_height = monitor.width, monitor.height
        
        hand_landmarker.detect_async(frame)
        try:
            if hand_landmarker.result and hand_landmarker.result.hand_landmarks and len(hand_landmarker.result.hand_landmarks[0]) == 21:
                pointer_tip = hand_landmarker.result.hand_landmarks[0][8]
                thumb_tip = hand_landmarker.result.hand_landmarks[0][4]
                
                distance = math.dist(
                    [thumb_tip.x, thumb_tip.y, thumb_tip.z],
                    [pointer_tip.x, pointer_tip.y, pointer_tip.z]
                )
                should_click = distance < 0.08 and (time.time() - last_click) > 0.5
                
                norm_x, norm_y = pointer_tip.x, pointer_tip.y
                mapped_x = map_coordinate(pointer_tip.x)
                mapped_y = map_coordinate(pointer_tip.y)
                pixel_x = int(norm_x * width)
                pixel_y = int(norm_y * height)
                monitor_x = int(mapped_x * monitor_width)
                monitor_y = int(mapped_y * monitor_height)
                x_array.append(monitor_x)
                y_array.append(monitor_y)
                cv2.circle(frame, (pixel_x, pixel_y), 10, (0, 255, 0), -1)
                
                thumb_pixel_x = int(thumb_tip.x * width)
                thumb_pixel_y = int(thumb_tip.y * height)
                cv2.circle(frame, (thumb_pixel_x, thumb_pixel_y), 10, (255, 0, 0), -1)
                
                cv2.line(frame, (pixel_x, pixel_y), (thumb_pixel_x, thumb_pixel_y), (0, 0, 255), 2)
                
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
                    if should_click:
                        pynput_mouse.click(Button.left, 1)
                        last_click = time.time()
                except RuntimeError:
                    pass
        except (AttributeError, IndexError):
            pass
        
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord("q"):
            break
    
    hand_landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()