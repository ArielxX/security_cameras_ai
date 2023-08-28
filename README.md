# Multi-Functional Web App: Object Detection, Face Recognition, and License Plate Recognition

This application includes three different functionalities: object detection, face recognition, and license plate recognition. The application uses Flask for the web interface and integrates the three functionalities.

### Object Detection

This application uses the YOLOv7 model for object detection. YOLOv7 is developed by Ultralytics and can be found [here](https://github.com/WongKinYiu/yolov7). The application allows users to upload an image and detect objects within it. It can be used independetly from the app by cloning the repository and running the following commands:

```bash
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
python detect.py --weights {weights_path} --source {image_path} --class 0 2 3 5 7 --project static --name detected_images --exist-ok --conf-thres 0.5 --iou-thres 0.6 --save-txt
```

### Face Recognition

This application uses a Siamese Neural Network for face recognition. The model is trained using the [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) dataset. The application allows users to upload an image and detect faces within it. It can be used independetly from the app by cloning the repository and running the following commands in python:

```python
from face_detect import predict
predict(photo_data)
```

Note that you should train your own model using your personal pictures for positive samples and test using the same kind of camera.

### License Plate Recognition

This application uses the Tessaract OCR engine for license plate recognition. The application allows users to upload an image and detect license plates within it. It can be used independetly from the app by cloning the repository and running the following commands in python:

```python
from plate_recognition import get_plate_number
get_plate_number(image_path)
```

Note that you will need to install the Tessaract OCR engine and the pytesseract library. For more information, visit the [pytesseract](https://tesseract-ocr.github.io/tessdoc/Installation.html) documentation.

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

4. **Install the Tesseract OCR Engine:**

    ```bash
    sudo apt install tesseract-ocr
    ```

    Note that this is for Linux. For other operating systems, visit the [pytesseract](https://tesseract-ocr.github.io/tessdoc/Installation.html) documentation.


5. **Run the Application:**

    ```bash
    cd ../app
    python app.py
    ```

    The application will be available at `http://127.0.0.1:5000` in your web browser.

## Requirements

- Python 3.6+
- Flask
- PyTesseract
- Other dependencies listed in `requirements.txt`

## Credits

This application uses the YOLOv7 model for object detection. YOLOv7 is developed by Ultralytics and can be found [here](https://github.com/ultralytics/yolov7).

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
