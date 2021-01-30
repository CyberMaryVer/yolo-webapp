print('Beginning of weights download...')

import cv2
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from flask import Flask, request, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename
import time
import boto3
import botocore
from skimage import io
from PIL import Image
# from amazon import *

BUCKET_NAME = 'yoloweights' # bucket name
KEY = 'yolov3.weights' # object key

s3 = boto3.resource('s3')
# s3 = boto3.resource('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

try:
    s3.Bucket(BUCKET_NAME).download_file(KEY, 'yolov.weights')
except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] == "404":
        print("The object does not exist.")
    else:
        raise

filename0 = 'cococlasses.sav'
c_classes = pickle.load(open(filename0, 'rb'))

def png2rgb(png, background=(255,255,255) ):
    """Image converting in case if we get a link"""
    image_np = png
    row, col, ch = image_np.shape

    if ch == 3:
        return image_np

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = image_np[:,:,0], image_np[:,:,1], image_np[:,:,2], image_np[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype='uint8')

def output_coordinates_to_box_coordinates(img_h, img_w, cx, cy, w, h):
    abs_x = int((cx - w / 2) * img_w)
    abs_y = int((cy - h / 2) * img_h)
    abs_w = int(w * img_w)
    abs_h = int(h * img_h)
    return abs_x, abs_y, abs_w, abs_h


def frame_color(coco, cnames='random', l=80):
    arr = np.random.randint(0, 255, (80, 3))
    if cnames == 'random':
        colors = [(int(x[0]), int(x[1]), int(x[2])) for x in arr]
    else:
        colors_arr = {x[0]: x[1] for x in zip(coco, arr)}
        colors = [(int(colors_arr[x][0]), int(colors_arr[x][1]),
                   int(colors_arr[x][2])) for x in cnames]
    return colors


def save_img(img, filename):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('static/images/'+filename, img)
    # cv2.imwrite(os.path.join('images/', filename), img)


def plot_img(img, saveimg=True, showplot=False):
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB));
    plt.axis('off');
    if saveimg:
        plt.savefig('images/yolo.png')
    if showplot:
        plt.show()


def main(img, net, filename, cococlasses=c_classes, precision=.4, multicolor=False):
    img_h, img_w = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_names = net.getUnconnectedOutLayersNames()
    large, medium, small = net.forward(output_names)

    all_outputs = np.vstack((large, medium, small))
    mask = np.where(all_outputs[:, 4] >= 0.1)
    objs = all_outputs[mask]

    boxes = [output_coordinates_to_box_coordinates(img_h, img_w, *x) for x in objs[:, :4]]
    confidences = objs[:, 4].astype(float)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, precision, 0.4)
    try:
        indices.flatten()
    except:
        print('No objects found. Try to reduce precision parameter')
        save_img(img, filename)
        return False

    class_names = [cococlasses[x.argmax()] for x in objs[:, 5:]]
    if multicolor:
        colors = frame_color(cococlasses)
    else:
        colors = frame_color(cococlasses, class_names)

    img_yolo = img.copy()

    for i in indices.flatten():
        x, y, w, h = boxes[i]
        class_name = class_names[i]
        confidence = confidences[i]
        color = colors[i]
        text = f'{class_name} {confidence:.3}'
        cv2.rectangle(img_yolo, (x, y), (x + w, y + h), color, 5)
        cv2.putText(
            img_yolo,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color)

    save_img(img_yolo, filename)
    return True

app = Flask(__name__)

net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# test = cv2.imread('templates/chess.jpg', cv2.IMREAD_UNCHANGED)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = "secret key"


@app.route('/')
def home():
    # return render_template('test.html', filename='static/images/yolo.png')
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # deep learning
        try:
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img_inp = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img_inp = png2rgb(img_inp)
            r = main(img_inp, net, filename, multicolor=False, precision=.4)
            time.sleep(4)
            if r:
                flash('Image successfully uploaded and recognized')
            else:
                flash('There are no objects known to the neural network in the image')
        except Exception as ex:
            flash('Unknown error with the image\n')
            print(ex)
        return render_template('upload.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    flash('You can save image by clicking on it')
    time.sleep(4)
    # print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='images/' + filename), code=301)

@app.route('/test/', methods=['GET'])
def respond():
    # Retrieve the name from url parameter
    name = request.args.get("name", None)

    # For debugging
    print(f"Image url {name}")

    response = {}

    # Check if user sent an url
    if not name:
        response["ERROR"] = "no url found"
    # Valid url
    else:
        try:
            image_np = io.imread(name)
            print(name)
            img_inp = png2rgb(image_np)
            print(img_inp.shape)
            r = main(img_inp, net, 'web_test.jpg', multicolor=False, precision=.4)
            if not r:
                response["MESSAGE"] = "No objects found. Try to reduce precision parameter"
                return response
            time.sleep(4)
            img_out = Image.open("static/images/web_test.jpg")
            img_out = img_out.getdata()
            return {"image": img_out}
        except Exception as ex:
            response["MESSAGE"] = f"Url {name}, {type(name)} is invalid"
            print(ex)
            return response


if __name__ == '__main__':

    port = os.environ.get('PORT')
    if port:
        # 'PORT' variable exists - running on Heroku, listen on external IP and on given by Heroku port
        app.run(host='0.0.0.0', port=int(port))

    else:
        # 'PORT' variable doesn't exist, running not on Heroku, presumably running locally, run with default
        #   values for Flask (listening only on localhost on default Flask port)
        app.run()
