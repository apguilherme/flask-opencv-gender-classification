from flask import render_template, request
import os
import cv2 # pip install opencv-python #in case of error
from PIL import Image
import pickle

IMG_PATH = "flask-opencv-gender-classification-main/static"
print(os.getcwd())
# load model
haar = cv2.CascadeClassifier("./flask-opencv-gender-classification-main/model/haarcascade_frontalface_default.xml")
# pickle files
mean = pickle.load(open("./flask-opencv-gender-classification-main/model/mean_preprocess.pickle", "rb"))
model_svm = pickle.load(open("./flask-opencv-gender-classification-main/model/model_svm.pickle", "rb"))
model_pca = pickle.load(open("./flask-opencv-gender-classification-main/model/pca50.pickle", "rb"))
print("Model loaded sucessfully")


def base():
    return render_template("base.html")

def index():
    if request.method == "POST":
        if request.files['file'].filename != '':
            file = request.files["file"]
            path = os.path.join(IMG_PATH, "uploads", file.filename)
            file.save(path)
            # processing
            w = get_width(path)
            pipeline_ML(path, file.filename, "bgr")
            print("File saved sucessfully in:", path)
            return render_template("index.html", uploaded=True, filename=file.filename, w=w)
    return render_template("index.html", uploaded=False)

def get_width(path):
    img = Image.open(path)
    size = img.size
    aspect = size[0]/size[1]
    w = 300*aspect
    return int(w)

def pipeline_ML(path, filename, color="bgr"):
    img = cv2.imread(path)
    # convert into gray scale
    if color == "bgr":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # crop face using haar cascade classifier
    faces = haar.detectMultiScale(gray, 1.5, 3)
    print("faces positions:", faces) # x, y, width, height
    gender_pred = ["Male", "Female"]
    for x, y, w, h in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2) # drawing rectangle
        roi = gray[y:y+h, x:x+w] # crop img
        roi = roi/255.0 # normalization 0-1
        if roi.shape[1] > 100:
            roi_resize = cv2.resize(roi, (100,100), cv2.INTER_AREA)
        else:
            roi_resize = cv2.resize(roi, (100,100), cv2.INTER_CUBIC)
        roi_reshape = roi_resize.reshape(1, 10000) # flattening 1, -1
        roi_mean = roi_reshape - mean # sutract from mean
        eig_img = model_pca.transform(roi_mean) # get eigen img
        results = model_svm.predict_proba(eig_img)[0] # pass to ML model SVM
        predict = results.argmax() # 0 or 1
        score = results[predict]
        text = "%s: %0.2f"%(gender_pred[predict], score)
        cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    cv2.imwrite(os.path.join(IMG_PATH, "predictions", filename), img)

