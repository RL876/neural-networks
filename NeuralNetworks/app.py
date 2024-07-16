import numpy as np
from NeuralNetworks.nn import models
from flask_ngrok import run_with_ngrok
from NeuralNetworks.utils import parseArgs
from flask import Flask, render_template, request

args = parseArgs()

global RESULT
global SCORES
RESULT = ""
SCORES = {}

# load model
model = models.load(args.model_dir)
labels = str(list(model.labels.values()))
# set app
app = Flask(__name__, template_folder="templates/")
if args.mode == "ngrok":
    run_with_ngrok(app)

@app.route("/", methods=["GET", "POST"])
def index():
    global RESULT
    global SCORES
    if request.method == "GET":
        return render_template("index.html", labels=labels, result=RESULT, scores=SCORES)
    if request.method == "POST":
        # get data
        data = getDataFromCanvas()
        # infer
        RESULT, SCORES = model(data=data)
        SCORES = dict(
            sorted(SCORES.items(), key=lambda item: item[1], reverse=True)[:3])
        return render_template("index.html", labels=labels, result=RESULT, scores=SCORES)

def getDataFromCanvas():
    data = request.form["canvas_data"]
    data = [float(i) for i in data.strip("[]").split(",")]
    data = np.resize(np.array(data), (280, 280, 1))
    data = np.array(data[::10, ::10, :])
    data = np.where(data > 254, 0, 1)
    data = data.astype(np.int32)
    return data
