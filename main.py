import cv2
import time
import math
import mediapipe as mp
import numpy as np
import pyautogui
from mediapipe.framework.formats import landmark_pb2
from screeninfo import get_monitors

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

def draw_landmarks(rgb_image, detection_result: mp.tasks.vision.HandLandmarkerResult):
   """Courtesy of https://github.com/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb"""
   try:
        if detection_result.hand_landmarks == []:
            return rgb_image
        else:
            hand_landmarks_list = detection_result.hand_landmarks
            annotated_image = np.copy(rgb_image)

            for idx in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[idx]
                
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
                mp.solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style())
            return annotated_image
   except:
        return rgb_image

def main():
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    monitor = get_monitors()[0]

    hand_landmarker = Landmarker()

    mouse_down = False

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        monitor_width, monitor_height = monitor.width, monitor.height
        
        hand_landmarker.detect_async(frame)
        #frame = draw_landmarks(frame, hand_landmarker.result)
        try:
            if hand_landmarker.result and hand_landmarker.result.hand_landmarks and len(hand_landmarker.result.hand_landmarks[0]) == 21:
                pointer_tip = hand_landmarker.result.hand_landmarks[0][8]
                thumb_tip = hand_landmarker.result.hand_landmarks[0][4]
                
                distance = math.sqrt((thumb_tip.x - pointer_tip.x)**2 + (thumb_tip.y - pointer_tip.y)**2 + (thumb_tip.z - pointer_tip.z)**2)
                should_click = distance < 0.06
                if should_click and not mouse_down:
                    pyautogui.mouseDown()
                    mouse_down = True
                elif not should_click and mouse_down:
                    pyautogui.mouseUp()
                    mouse_down = False
                
                norm_x, norm_y = pointer_tip.x, pointer_tip.y
                pixel_x = int(norm_x * width)
                pixel_y = int(norm_y * height)
                monitor_x = int(norm_x * monitor_width)
                monitor_y = int(norm_y * monitor_height)
                cv2.circle(frame, (pixel_x, pixel_y), 10, (0, 255, 0), -1)
                
                thumb_pixel_x = int(thumb_tip.x * width)
                thumb_pixel_y = int(thumb_tip.y * height)
                cv2.circle(frame, (thumb_pixel_x, thumb_pixel_y), 10, (255, 0, 0), -1)
                
                cv2.line(frame, (pixel_x, pixel_y), (thumb_pixel_x, thumb_pixel_y), (0, 0, 255), 2)
                
                pyautogui.moveTo(monitor_x, monitor_y)
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