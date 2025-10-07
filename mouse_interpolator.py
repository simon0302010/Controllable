import time
import threading
import queue
import pyautogui

class MouseInterpolator:
    def __init__(self, steps=10, sleep_time=0.005) -> None:
        pyautogui.PAUSE = 0
        pyautogui.FAILSAFE = False
        self.target_queue = queue.Queue(maxsize=1)
        self.running = True
        self.steps = steps
        self.sleep_time = sleep_time
        self.interpolation_thread = threading.Thread(target=self._interpolate_movement, daemon=True)
        self.interpolation_thread.start()

    def _interpolate_movement(self):
        current_x, current_y = pyautogui.position()
        while self.running:
            try:
                target_x, target_y = self.target_queue.get(timeout=0.01)
                for step in range(1, self.steps + 1):
                    if not self.running:
                        break
                    
                    t = step / self.steps
                    t = 1- (1 - t) ** 2
                    interp_x = current_x + (target_x - current_x) * t
                    interp_y = current_y + (target_y - current_y) * t
                    try:
                        pyautogui.moveTo(int(interp_x), int(interp_y), _pause=False)
                    except RuntimeError:
                        pass
                    time.sleep(self.sleep_time)
                current_x, current_y = target_x, target_y
            except queue.Empty:
                current_x, current_y = pyautogui.position()
                
    def move_to(self, x, y):
        try:
            self.target_queue.get_nowait()
        except queue.Empty:
            pass
        self.target_queue.put((x, y))
        
    def stop(self):
        self.running = False
        if self.interpolation_thread.is_alive():
            self.interpolation_thread.join(timeout=1.0)