from flask import Flask, render_template, request, redirect, url_for
import subprocess

app = Flask(__name__)

labels_map = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorbike",
    4: "aeroplane",
    5: "bus",
    6: "train",
    7: "truck"
}

def run_detection(weights_path, image_path):
    command = f"python ../yolov7/detect.py --weights {weights_path} --source {image_path} --class 0 2 3 5 7 --project static --name detected_images --exist-ok --conf-thres 0.5 --iou-thres 0.6 --save-txt"
    subprocess.run(command, shell=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)

        image = request.files['image']

        if image.filename == '':
            return redirect(request.url)

        if image:
            image_path = f"static/uploaded_images/{image.filename}"
            image.save(image_path)

            weights_path = "../yolov7/yolov7-tiny.pt"
            run_detection(weights_path, image_path)

            labels = {}

            labels_path = f"static/detected_images/labels/{image.filename.split('.')[0]}.txt"
            detection_result_path = f"static/detected_images/{image.filename}"

            # check if the detection result exists
            try:
                with open(detection_result_path, 'rb') as f:
                    pass

                # read the labels file
                try:
                    with open(labels_path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            # split the line by space
                            line = line.split(' ')
                            # get the class id
                            class_id = int(line[0])
                            if class_id in labels_map:
                                labels[labels_map[class_id]] = labels.get(labels_map[class_id], 0) + 1
                except FileNotFoundError:
                    print("No labels file found")

            except FileNotFoundError:
                print("No detection result found")
                detection_result_path = image_path

            # remove the labels file
            subprocess.run(f"rm {labels_path}", shell=True)

            return render_template('index.html', detection_result=detection_result_path, labels=labels)

    return render_template('index.html', detection_result=None)

if __name__ == '__main__':
    app.run(debug=True)
