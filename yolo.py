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
from urllib.request import urlretrieve
from PIL import Image

# from amazon import *
# s3 = boto3.resource('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

BUCKET_NAME = 'yoloweights' # bucket name
KEY = 'yolov3.weights' # object key

# for heroku
s3 = boto3.resource('s3')

try:
    s3.Bucket(BUCKET_NAME).download_file(KEY, 'yolov.weights')
except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] == "404":
        print("The object does not exist.")
    else:
        raise

filename0 = 'cococlasses.sav'
c_classes = pickle.load(open(filename0, 'rb'))
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = "secret key"

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def resize_img(img, resizing=600):
    h, w, d = img.shape
    inv_coef = 1
    if w > resizing:
        coef = h/w
        wn = resizing
        hn = int(np.round(wn*coef))
    inv_coef = w/wn
    img = cv2.resize(img, (wn, hn))
    return img, inv_coef


def png2rgb(png, background=(255,255,255) ):
    """Image converting in case if we get a link"""
    image_np = png
    row, col, ch = image_np.shape

    if ch == 3:
        return image_np

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


def frame_color(coco, cnames=None, l=80):
    arr = np.random.randint(0, 255, (80, 3))
    if cnames == None:
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


def main(img, net, filename, cococlasses=c_classes, precision=.4, high_quality=False):
    if high_quality:
        new_img = img
    else:
        new_img = None

    img, resize_coef = resize_img(img, 600)
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
    colors = frame_color(cococlasses, class_names)

    # use the initial high resolution image to draw frames or resize the current image
    if new_img is not None:
        img_yolo = new_img
    else:
        img_yolo = img.copy()
        # resize_coef = 1.5
        img_yolo = cv2.resize(img_yolo, dsize=None, fx=resize_coef,fy=resize_coef)

    for i in indices.flatten():
        x, y, w, h = [int(crd * resize_coef) for crd in boxes[i]]
        # print(x, y, w, h)
        class_name = class_names[i]
        confidence = confidences[i]
        color = colors[i]
        text = f'{class_name} {confidence:.2}'

        # print(class_name)
        if class_name == 'person':
            center_coord = (x+w//2, y+h//2)
            pt1 = (center_coord[0] - 20, center_coord[1])
            pt2 = (center_coord[0] + 20, center_coord[1])
            pt3 = (center_coord[0], center_coord[1] - 20)
            pt4 = (center_coord[0], center_coord[1] + 20)
            color = (0,255,0)

            # Draw transparent frames
            sub_img = img_yolo[y:y + h, x:x + w]
            white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
            res = cv2.addWeighted(sub_img, 0.75, white_rect, 0.25, 1.0)

            # Putting the image back to its position
            img_yolo[y:y + h, x:x + w] = res

            # Draw green circle
            img_yolo = cv2.circle(img_yolo, center_coord, 16, color, 1)
            img_yolo = cv2.circle(img_yolo, center_coord, 12, color, 1)
            img_yolo = cv2.line(img_yolo, pt1, pt2, color, 1)
            img_yolo = cv2.line(img_yolo, pt3, pt4, color, 1)

        cv2.rectangle(img_yolo, (x, y), (x + w, y + h), color, 3)
        cv2.putText(
            img_yolo,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_COMPLEX,
            0.4,
            color)

    save_img(img_yolo, filename)
    return True

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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
            # img_inp, coef = resize_img(img_inp, 600)
            r = main(img_inp, net, filename, precision=.5)
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
    # time.sleep(4)
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
            testpath = "static/images/web_test.jpg"
            urlretrieve(name, testpath)
            img_inp = cv2.imread(testpath, cv2.IMREAD_UNCHANGED)
            print(name)
            img_inp = png2rgb(img_inp)
            # img_inp, coef = resize_img(img_inp, 600)
            img_shape = img_inp.shape
            print(img_shape)
            r = main(img_inp, net, 'web_test.jpg', precision=.4)
            if not r:
                response["MESSAGE"] = "No objects found. Try to reduce precision parameter"
                return response
            time.sleep(2)
            img_out = Image.open("static/images/web_test.jpg")
            img_out = np.array(img_out.getdata()).tolist()
            return {"image": img_out, "shape": img_shape}
        except Exception as ex:
            response["MESSAGE"] = f"Url {name} is invalid"
            print(ex)
            return response


if __name__ == '__main__':
    # test = cv2.imread('templates/chess.jpg', cv2.IMREAD_UNCHANGED)

    port = os.environ.get('PORT')
    if port:
        # 'PORT' variable exists - running on Heroku, listen on external IP and on given by Heroku port
        app.run(host='0.0.0.0', port=int(port))

    else:
        # 'PORT' variable doesn't exist, running not on Heroku, presumably running locally, run with default
        #   values for Flask (listening only on localhost on default Flask port)
        app.run()
