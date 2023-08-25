# Object Detection Web App

This is a simple web application for object detection using YOLOv7. It allows users to upload an image and detect objects within it. The application uses Flask for the web interface and integrates with YOLOv7 for object detection.

## How to Use

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/ArielxX/security_cameras_ai.git
    cd object-detection-app
    ```

2. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Download the YOLOv7 Model:**

    ```bash
    git clone https://github.com/ultralytics/yolov7.git
    cd yolov7
    pip install -r requirements.txt
    ```

3. **Run the Application:**

    ```bash
    cd ../app
    python app.py
    ```

    The application will be available at `http://127.0.0.1:5000` in your web browser.

4. **Upload an Image:**

    - Click on the "Choose File" button to upload an image.
    - After selecting the image, click the "Detect Objects" button.
    - The application will display the uploaded image along with detected object labels (if any).

5. **Exiting the Application:**

    Press `Ctrl+C` in the terminal where the application is running to stop the server.

## Requirements

- Python 3.6+
- Flask
- Other dependencies listed in `requirements.txt`

## Credits

This application uses the YOLOv7 model for object detection. YOLOv7 is developed by Ultralytics and can be found [here](https://github.com/ultralytics/yolov7).

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
