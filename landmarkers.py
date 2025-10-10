import mediapipe as mp

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
            min_hand_detection_confidence = 0.5,
            min_hand_presence_confidence = 0.5,
            min_tracking_confidence = 0.5,
            result_callback=update_result
        )
        
        self.landmarker = self.landmarker.create_from_options(options)
        
    def detect_async(self, frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.timestamp += 1
        self.landmarker.detect_async(image = mp_image, timestamp_ms=self.timestamp)
        
    def close(self):
        self.landmarker.close()