import os
import glob
import awsgi
import argparse
import numpy as np
from urllib.request import urlopen
from flask import Flask, render_template, request


global RESULT
global SCORES
RESULT = "draw something!"
SCORES = {}


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", action="store_true")
    return parser.parse_args()


def train():
    # load data
    x = np.empty((0, 28, 28, 1), int)
    y = np.empty((0), int)
    for fname in glob.glob(os.path.join("..", "dataset", "*.npy")):
        label = os.path.splitext(os.path.basename(fname))[0]
        data = np.load(fname)
        data = data.reshape(len(data), 28, 28, 1)
        data = data[:100000]
        x = np.append(x, data, axis=0)
        y = np.append(y, np.repeat(label, 100000))
    x = x / 255
    x = x.reshape(x.shape[0], np.prod(np.array(x.shape[1:])))
    # one hot encoding
    labels = np.unique(y)
    labels.sort()
    label_dict = dict()
    for value in labels:
        key = np.where(labels == value)[0][0]
        label_dict[key] = str(value)
        y = np.where(y == value, key, y)
    y = y.astype(int)
    y = np.eye(len(np.unique(y)))[y].astype(int)
    # dataset split
    row = len(y)
    randomize = np.arange(row)
    np.random.shuffle(randomize)
    x = x[randomize]
    y = y[randomize]
    train_index = int(row * 0.9)
    test_index = train_index + int(row * 0.09)
    X_train = x[:train_index]
    X_test = x[train_index:test_index]
    Y_train = y[:train_index]
    Y_test = y[train_index:test_index]
    # activation
    def relu(z):
        return np.maximum(0, z)
    def relu_backward(dz):
        return 1 * (dz > 0)
    def softmax(z):
        z = z - np.max(z, axis=1, keepdims=True)
        z_exp = np.exp(z)
        y_hat = z_exp / np.sum(z_exp, axis=1, keepdims=True)
        y_hat = np.where(y_hat == 0, 10**-10, y_hat)
        return y_hat
    def softmax_backward(y_hat, Y):
        return y_hat - Y
    # hyperparameters
    lr = 1e-3
    epochs = 300
    batch_size = 16
    weight_scale = 1e-1
    # init w & b
    W1 = np.random.randn(784, 64) * weight_scale
    b1 = np.random.randn(64) * weight_scale
    W2 = np.random.randn(64, 16) * weight_scale
    b2 = np.random.randn(16) * weight_scale
    W3 = np.random.randn(16, 3) * weight_scale
    b3 = np.random.randn(3) * weight_scale
    # batch
    batch_iteration = int(X_train.shape[0] / batch_size)
    batch_iteration = batch_iteration + 1 if X_train.shape[0] % batch_size > 0 else batch_iteration
    # start train
    best_val_acc = np.load("model.npz")["val_acc"] if os.path.exists("model.npz") else 0
    for i in range(epochs):
        loss = acc = val_loss = val_acc = 0
        for j in range(batch_iteration):
            batch_X = X_train[j*batch_size:(j+1)*batch_size]
            batch_Y = Y_train[j*batch_size:(j+1)*batch_size]
            m = batch_X.shape[0]
            # forward propogation
            Z1 = np.dot(batch_X, W1) + b1
            X1 = relu(Z1)
            Z2 = np.dot(X1, W2) + b2
            X2 = relu(Z2)
            Z3 = np.dot(X2, W3) + b3
            Y_hat = softmax(Z3)
            # batch loss, acc
            batch_loss = np.average(np.square(Y_hat - batch_Y) ** 0.5)
            batch_acc = (np.argmax(Y_hat, axis=1) == np.argmax(batch_Y, axis=1)).sum() / m
            loss = loss + batch_loss / batch_iteration
            acc = acc + batch_acc / batch_iteration
            # backward propogation
            # layer 3
            dZ3 = softmax_backward(Y_hat, batch_Y)
            dW3 = np.dot(X2.T, dZ3) / m
            db3 = np.sum(dZ3, axis=0) / m
            # layer 2
            dZ2 = np.dot(dZ3, W3.T) * relu_backward(Z2)
            dW2 = np.dot(X1.T, dZ2) / m
            db2 = np.sum(dZ2, axis=0) / m
            # layer 1
            dZ1 = np.dot(dZ2, W2.T) * relu_backward(Z1)
            dW1 = np.dot(batch_X.T, dZ1) / m
            db1 = np.sum(dZ1, axis=0) / m
            # gradient descent
            W1 = W1 - lr * dW1
            W2 = W2 - lr * dW2
            W3 = W3 - lr * dW3
            b1 = b1 - lr * db1
            b2 = b2 - lr * db2
            b3 = b3 - lr * db3
        # loss, acc, val_loss, val_acc
        Z1 = np.dot(X_test, W1) + b1
        X1 = relu(Z1)
        Z2 = np.dot(X1, W2) + b2
        X2 = relu(Z2)
        Z3 = np.dot(X2, W3) + b3
        Y_hat = softmax(Z3)
        val_loss = np.average(np.square(Y_hat - Y_test) ** 0.5)
        val_acc = (np.argmax(Y_hat, axis=1) == np.argmax(Y_test, axis=1)).sum() / Y_test.shape[0]
        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            np.savez("model.npz", W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3, labels=labels, val_acc=best_val_acc)
        print("Epoch {: 5d}/{}\t- loss: {:.4f} - acc: {:.4f} - val_loss: {:.4f} - val_acc: {:.4f}".format(i+1, epochs, loss, acc, val_loss, val_acc))
    print("Training Complete: val_acc = {:.4f}".format(best_val_acc))


# load model
model = np.load("model.npz") if os.path.exists("model.npz") else None
labels = None
if model:
    W1 = model["W1"]
    b1 = model["b1"]
    W2 = model["W2"]
    b2 = model["b2"]
    W3 = model["W3"]
    b3 = model["b3"]
    labels_dict = model["labels"]
    labels = " / ".join(labels_dict)
def relu(z):
    return np.maximum(0, z)
def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    z_exp = np.exp(z)
    y_hat = z_exp / np.sum(z_exp, axis=1, keepdims=True)
    y_hat = np.where(y_hat == 0, 10**-10, y_hat)
    return y_hat
# set app
app = Flask("doodle", template_folder="./")
@app.route("/doodle", methods=["GET", "POST"])
def index():
    global RESULT
    global SCORES
    if not model:
        RESULT = "No Model!"
    try:
        if request.method == "POST":
            # get data
            with urlopen(request.form["img"]) as response:
                img = response.read()
            img = np.frombuffer(img, np.uint8)
            img = np.resize(np.array(img), (280, 280, 1))
            img = img[:280, :280].reshape(28, 10, 28, 10).max(axis=(1, 3))
            img = np.where(img > 254, 0, 1)
            img = img.astype(np.int32)
            img = img.reshape(np.prod(np.array(img.shape)))
            img = np.expand_dims(img, axis=0)
            # infer
            Z1 = np.dot(img, W1) + b1
            X1 = relu(Z1)
            Z2 = np.dot(X1, W2) + b2
            X2 = relu(Z2)
            Z3 = np.dot(X2, W3) + b3
            Y_hat = softmax(Z3)[0]
            RESULT = str(labels_dict[np.argmax(Y_hat)])
            for i, label in enumerate(labels_dict):
                SCORES[str(label)] = "{:.3f}".format(Y_hat[i])
            SCORES = dict(sorted(SCORES.items(), key=lambda item: item[1], reverse=True)[:3])
    except BaseException as e:
        RESULT = f"python except: {str(e)}"
    return render_template("index.html", labels=labels, result=RESULT, scores=SCORES)


def lambda_handler(event, context):
    return awsgi.response(app, event, context)
    

if __name__ == "__main__":
    args = parseArgs()
    train() if args.train else None
    app.run(host="0.0.0.0", port=5000)