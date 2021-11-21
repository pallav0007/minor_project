from flask_cors import CORS
from flask import Flask, request, render_template, json, jsonify, send_from_directory,send_file
import json
import cv2
import numpy as np
import io
from base64 import b64encode
from preprocessing import detect_hand,image_preprocessing
from PIL import Image
# from flask_ngrok import run_with_ngrok
app = Flask(__name__,static_url_path="", static_folder="static/frontend")
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app)
# run_with_ngrok(app)

@app.route("/base", methods=["GET"])
def base():
    return render_template('base.html')
@app.route("/real_time", methods=["GET"])
def rt():
    return render_template('real_time.html')
@app.route("/real_time_video", methods=["GET"])
def rt_video():
    return render_template('real_time_video.html')
@app.route("/", methods=["GET"])
def main():
    return render_template('index.html')
@app.route("/api/prepare/realtime", methods=["POST"])
def prepare_Rt():
    file = request.files['file']
    # file=np.array(file)
    res = preprocessing(file)
    return json.dumps({"image": res.tolist()})

@app.route("/api/predict", methods=['POST'])
def make_predictions():
    file = request.get_json(force=True)
    file=file["image"]
    res,c = preprocessing(file)
    return json.dumps({"image": res.tolist(),"bbox":c.tolist()})

@app.route("/api/prepare", methods=["POST"])
def prepare():
    file = request.files['file']
    res,c= preprocessing(file)
    file_object = io.BytesIO()
    img = Image.fromarray(c.astype('uint8'))
    img.save(file_object, 'PNG')
    base64img = "data:image/png;base64," + b64encode(file_object.getvalue()).decode('ascii')
    # print(base64img)
    return json.dumps({"image": res.tolist(),"bbox":base64img})
@app.route('/image.png')
def image():
    # my numpy array
    raw_data=cv2.imread("raw_data.jpg")
    arr = np.array(raw_data)

    # convert numpy array to PIL Image
    img = Image.fromarray(arr.astype('uint8'))

    # create file-object in memory
    file_object = io.BytesIO()

    # write PNG in file-object
    img.save(file_object, 'PNG')

    # move to beginning of file so `send_file()` it will read from start
    file_object.seek(0)

    return send_file(file_object, mimetype='image/PNG')


@app.route('/model')
def model():
    json_data = json.load(open("./model_js/model.json"))
    return jsonify(json_data)
@app.route("/group1-shard1of4.bin")
def load_shards_one():
    return send_from_directory('model_js',"group1-shard1of4.bin")
@app.route("/group1-shard2of4.bin")
def load_shards_two():
    return send_from_directory('model_js',"group1-shard2of4.bin")
@app.route("/group1-shard3of4.bin")
def load_shards_three():
    return send_from_directory('model_js',"group1-shard3of4.bin")
@app.route("/group1-shard4of4.bin")
def load_shards_four():
    return send_from_directory('model_js',"group1-shard4of4.bin")
# @app.route('/<path:path>')
# def load_shards(path):
#     return send_from_directory('model_js', path)


def default_preprocessing(name):
    img=cv2.imread(name)
    res, c = detect_hand(img)
    # cv2.imwrite("raw_data.jpg", c)
    # cv2.imwrite("res.jpg", res)
    res = image_preprocessing(res)

    # res = cv2.resize(img, dsize=(100,100), interpolation=cv2.INTER_CUBIC)
    # res=res.reshape((100,100,1))
    # file.save("static/UPLOAD/img.png") # saving uploaded img
    # cv2.imwrite("static/UPLOAD/test.png", res) # saving processed image
    return res
def preprocessing(file):
    print(file)
    try:
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        # img=cv2.imread("pallav.jpg")

        res,c=detect_hand(img)
        # cv2.imwrite("raw_data.jpg",c)
        # cv2.imwrite("res.jpg", res)
        res=image_preprocessing(res)

        # res = cv2.resize(img, dsize=(100,100), interpolation=cv2.INTER_CUBIC)
        # res=res.reshape((100,100,1))
        # file.save("static/UPLOAD/img.png") # saving uploaded img
        # cv2.imwrite("static/UPLOAD/test.png", res) # saving processed image
        return res,c
    except:
        return np.array([]),np.array([])


if __name__ == "__main__":
    app.run()
