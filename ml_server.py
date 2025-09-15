import os
import torch
import cv2
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

from ultralytics import YOLO
import cv2




app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

WEIGHTS_PATH = "weights/best.pt"


def detect_objects(image_path, weights_path="weights/best.pt"):
    
    model = YOLO(weights_path)
    
   
    results = model.predict(source=image_path, imgsz=640, conf=0.5)
    
    
    img0 = cv2.imread(image_path)
    
    
    for result in results:
        boxes = result.boxes.xyxy
        confidences = result.boxes.conf
        classes = result.boxes.cls
        for xyxy, conf, cls in zip(boxes, confidences, classes):
            c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
            label = f"{int(cls)} {conf:.2f}"
            cv2.rectangle(img0, c1, c2, (255, 0, 0), 2)
            cv2.putText(img0, label, (c1[0], c1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    return img0



@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            result_img = detect_objects(filepath, WEIGHTS_PATH)

            result_path = os.path.join(app.config["RESULT_FOLDER"], filename)
            cv2.imwrite(result_path, result_img)

            return render_template("result.html", result_image= result_path)
        
    return render_template("upload.html")


@app.route('/blog')
def blog():
    return render_template('blog.html')  

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')


if __name__ == "__main__":
    app.run(debug=True)

