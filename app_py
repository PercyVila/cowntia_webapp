import base64
import io
import os

import cv2 as cv
import numpy as np
from PIL import Image

from flask import Flask, render_template, request, jsonify

INIT_TEMPLATE = 'init.html'
RESULT_TEMPLATE = 'results.html'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def np_array_2_uri(np_array):
    img = Image.fromarray(np_array.astype("uint8"))

    raw_bytes = io.BytesIO()
    img.save(raw_bytes, "JPEG")
    raw_bytes.seek(0)

    img_base64 = base64.b64encode(raw_bytes.getvalue()).decode('ascii')
    mime = "image/jpeg"
    uri = "data:%s;base64,%s" % (mime, img_base64)

    return uri


def parse_file_storage_2_numpy(file):
    # read image file string data
    file_str = file.read()
    # convert string data to numpy array
    np_img = np.frombuffer(file_str, np.uint8)
    # convert numpy array to image
    return cv.imdecode(np_img, cv.IMREAD_COLOR)


@app.route('/health-check')
def hello_world():
    return 'Online', 200


@app.route('/', methods=['GET', 'POST'])
def init_gui():
    return render_template('init.html')


@app.route('/process_img', methods=['POST'])
def process_img():

    size = int(request.form.get('number'))

    file_f = request.files['file_front']
    file_s = request.files['file_side']

    allowed_file_f = allowed_file(file_f.filename)
    allowed_file_s = allowed_file(file_s.filename)

    cond_1 = file_s and file_f
    cond_2 = allowed_file_s and allowed_file_f

    if cond_1 and cond_2:
        im_f = parse_file_storage_2_numpy(file_f)
        im_s = parse_file_storage_2_numpy(file_s)

        im_f_out, im_s_out, mass = process_images(im_f, im_s, size)

        im_input_uri = np_array_2_uri(cv.cvtColor(im_f_out, cv.COLOR_BGR2RGB))
        im_output_uri = np_array_2_uri(cv.cvtColor(im_s_out, cv.COLOR_BGR2RGB))

        return render_template('result.html',
                               mass=round(mass),
                               im_input=im_input_uri,
                               im_output=im_output_uri)


def process_images(im_f, im_s, size):
    # MOCK
    kg = 400
    return im_f, im_s, kg


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
