from NeuralNetworks.app import app
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
