from flask_cors import CORS
from flask import Flask, request, render_template, json, jsonify, send_from_directory
import json
import cv2
import numpy as np
import io
from TensorFlowJSClientSidePrediction.preprocessing import detect_hand,image_preprocessing
from flask_ngrok import run_with_ngrok
app = Flask(__name__)
CORS(app)
# run_with_ngrok(app)
@app.route("/base", methods=["GET"])
def base():
    return render_template('base.html')
@app.route("/real_time", methods=["GET"])
def rt():
    return render_template('real_time.html')
@app.route("/", methods=["GET"])
def main():
    return render_template('index.html')
@app.route("/api/prepare/realtime", methods=["POST"])
def prepare_Rt():
    file = request.files['file']
    # file=np.array(file)
    res = preprocessing(file)
    return json.dumps({"image": res.tolist()})

@app.route("/api/prepare", methods=["POST"])
def prepare():
    file = request.files['file']
    res = preprocessing(file)
    return json.dumps({"image": res.tolist()})


@app.route('/model')
def model():
    json_data = json.load(open("./model_js/model.json"))
    return jsonify(json_data)


@app.route('/<path:path>')
def load_shards(path):
    return send_from_directory('model_js', path)


def preprocessing(file):
    print(file)
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    # img=cv2.imread("pallav.jpg")

    res=detect_hand(img)
    cv2.imwrite("res.jpg", res)
    res=image_preprocessing(res)

    # res = cv2.resize(img, dsize=(100,100), interpolation=cv2.INTER_CUBIC)
    # res=res.reshape((100,100,1))
    # file.save("static/UPLOAD/img.png") # saving uploaded img
    # cv2.imwrite("static/UPLOAD/test.png", res) # saving processed image
    return res


if __name__ == "__main__":
    app.run()
