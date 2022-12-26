import numpy as np
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
from NeuralNetworks.utils import parseArgs, getDataset
from NeuralNetworks.nn import models, layers, optimizers, losses

args = parseArgs()

global RESULT
global SCORES
RESULT = ""
SCORES = {}


def train():
    # get data
    data = getDataset(args.datasets)
    input_shape = data.x_train[0].shape
    # create model
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(units=64, activation="Relu"))
    model.add(layers.Dense(units=16, activation="Relu"))
    model.add(layers.Dense(units=len(data.label_dict),
                           activation="Softmax"))
    model.summary()
    model.compile(optimizer=optimizers.AdaGrad(learning_rate=args.lr),
                  loss=losses.CategoricalCrossentropy(),
                  metrics=["accuracy"])
    model.fit(data.x_train,
              data.y_train,
              epochs=args.epochs,
              batch_size=args.batch,
              validation_data=(data.x_test, data.y_test))
    model.save(model_dir=args.model_dir, labels=data.label_dict)


def infer():
    # get data
    data = getDataset(args.datasets)
    # load model
    model = models.load(args.model_dir)
    # inferＦ
    results = model(data=data.x_val[0])
    print(results)


def runApp():
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

    # run app
    if args.mode == "dev":
        app.run(host="127.0.0.1", port=5000, debug=True)
    if args.mode == "ngrok":
        app.run()
    if args.mode == "ops":
        app.run(host="0.0.0.0", port=80)


def run():
    if args.mode == "train":
        train()
        infer()
    else:
        runApp()


if __name__ == "__main__":
    run()
