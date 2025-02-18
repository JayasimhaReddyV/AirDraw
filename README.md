# AirDraw
Air Draw is a gesture-controlled drawing application that enables users to draw, move, and erase using hand gestures. It eliminates the need for physical tools, allowing users to interact with a virtual canvas seamlessly.

Features
- Hand Gesture-Based Drawing – Use hand movements to draw on a virtual canvas.
- Multiple Modes – Switch between Draw, Move, and Erase modes using gestures.
- Color Selection – Choose different ink colors for drawing.
- Save Artwork – Press the 'S' key to save your drawing.
- Dual-Screen Interface – One screen displays the camera feed, and the other shows the drawing canvas.
- Exit Button – An on-screen exit button allows users to close the application easily.
- Optimized Camera Feed – Adjusted quality for better visibility.

Technologies Used
- Python
- OpenCV (for real-time image processing)
- Mediapipe (for hand tracking)
- NumPy

How It Works
- The application detects hand gestures using Mediapipe Hands.
- Users can switch between drawing, moving, and erasing using gestures.
- The drawing appears on a separate canvas while the live feed is displayed.
- Press 'S' to save the drawing.
