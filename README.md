# Controllable

A program to control your computer through its camera like magic ✨🪄.

## Demo

https://github.com/user-attachments/assets/beb703c2-fc6a-499f-8027-e6053bb562d0

## Requirements
- Python 3.9 - 3.12
- A camera connected to your computer

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/simon0302010/Controllable.git
    cd Controllable
    ```

2.  **Install the dependencies:**

    It is recommended to use a virtual environment.

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

    Install the required packages using `pip`:

    ```bash
    pip install -e . # or pip install controllable-py
    pip uninstall -y opencv-contrib-python opencv-python-headless
    pip cache purge
    pip install opencv-python-headless
    ```
    > `opencv-contrib-python` is being uninstalled to prevent issues with `PyQt5`.

## Usage

1.  **Run the application:**

    ```bash
    python -m controllable
    ```
    > You can also just run `controllable`

2.  **Calibrate:**

    A window will open showing your camera feed. Place your hand in a comfortable position in front of the camera and press the **"Calibrate"** button.

3.  **Start controlling:**

    After calibration, press the **"Start"** button. You can now control your mouse by moving your hand.
    To prevent accidental inputs, keep your hand the same distance from the camera during usage.

    -   **Mouse movement:** The position of your index finger controls the cursor.
    -   **Clicking:** Tap your index finger and thumb together to perform a click.
    -   **Dragging:** Hold your index finger and thumb together to drag (has to be enabled in settings)

4.  **Stop:**

    Press the **"Stop"** button to stop controlling the mouse.

## Known Issues
- Mouse control and clicking doesn't work under Wayland

## License
This project is licensed under the GNU General Public License v3.0. For more details, see the [LICENSE](LICENSE) file.
