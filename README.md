# 🎨 AI Air Canvas (Computer Vision HCI)

A contactless digital drawing interface built with **Python**, **OpenCV**, and **MediaPipe**. This project utilizes hand landmark detection to track finger movements in real-time, allowing users to draw on a virtual canvas using simple hand gestures.

## 🚀 Key Features
* **Hand Tracking Engine:** Uses MediaPipe to detect 21 hand landmarks with high precision.
* **Dynamic Brush Control:** Adjusts brush thickness based on the distance between the Index and Middle finger (Simulated Pressure Sensitivity).
* **Gesture State Machine:**
    * **☝ Index Up + Thumb Closed:** Draw Mode.
    * **✋ Index Up + Thumb Open:** Hover/Cursor Mode.
    * **👆 Selection:** Hover over UI elements to change colors/tools.
* **Signal Smoothing:** Implemented Exponential Moving Average (EMA) to reduce webcam jitter.
* **Performance:** Optimized for real-time usage (~30 FPS on standard CPU).

## 🛠 Tech Stack
* **Language:** Python 3.10
* **Computer Vision:** OpenCV (`cv2`), MediaPipe
* **Math/Logic:** NumPy, Coordinate Geometry

## ⚙️ How to Run
1.  Clone the repository:
    ```bash
    git clone [https://github.com/Rishwanth11/AI-Air-Canvas.git](https://github.com/Rishwanth11/AI-Air-Canvas.git)
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the application:
    ```bash
    python air_canvas.py
    ```

## 📸 Controls
* **'S' key:** Save your artwork.
* **'U' key:** Undo last stroke.
* **'Q' key:** Quit application.

---
*Developed by Rishwanth - ECE Student*
