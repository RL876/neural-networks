import os
import h5py
import numpy as np
from NeuralNetworks.nn import layers, optimizers, losses


class Sequential(object):
    def __init__(self):
        self.__layers = []
        self.__history = {
            "loss": [],
            "acc": [],
            "val_loss": [],
            "val_acc": [],
        }
        self.__history_best = {
            "loss": 0,
            "acc": 0,
            "val_loss": 0,
            "val_acc": 0,
        }

    def add(self, layer: layers.Layer):
        self.__layers.append(layer)

    def summary(self):
        # format
        format_summary = "{:>15}{:>30}{:>15}"
        # model
        print(f"Model: {self.__class__.__name__}")
        # layers
        print("-" * 60)
        print(format_summary.format(
            "Layer (type)", "Output Shape", "Param #"))
        print("=" * 60)
        param = 0
        for layer in self.__layers:
            layer_summary = layer.summary
            param = param + layer_summary["param"]
            print(format_summary.format(
                str(layer_summary["layer"]),
                str(layer_summary["output_shape"]),
                str(layer_summary["param"]),
            ))
        print("=" * 60)
        print(f"Total params: {param}")
        print("-" * 60)

    def compile(self, optimizer: optimizers.Optimizer, loss: losses.Loss, metrics: list):
        self.__optimizer = optimizer
        self.__loss = loss
        self.__metrics = metrics
        self.__imgsz = self.__layers[0].input_shape

    def fit(self,
            x_train: np.array,
            y_train: np.array,
            epochs: int,
            batch_size: int,
            validation_data: tuple
            ):
        # validation data
        x_test = validation_data[0]
        y_test = validation_data[1]
        # set batch
        if batch_size < 1 or batch_size > x_train.shape[0]:
            batch_size = x_train.shape[0]
        batch_iteration = x_train.shape[0] // batch_size
        # train
        if self.__layers[-1].output_shape[0] != y_train.shape[1]:
            raise ValueError(
                f"Last layer dimasion `{self.__layers[-1].output_shape[0]}` not match dataset dimension `{y_train.shape[1]}`"
            )
        print("-" * 60)
        print(f"Epochs:\t{epochs}")
        print(f"Batch:\t{batch_size}")
        print("=" * 60)
        for epoch in range(1, epochs + 1):
            # reset loss, acc, val_loss, val_acc
            loss = acc = val_loss = val_acc = 0
            for batch in range(batch_iteration):
                # get batch data
                batch_mask = np.random.choice(
                    x_train.shape[0], batch_size, replace=False
                )
                x_batch = x_train[batch_mask]
                y_batch = y_train[batch_mask]
                # sequence forward
                y_hat = self.__forward(x_batch)
                # batch loss, acc
                batch_acc, batch_loss = self.__loss(y_hat, y_batch)
                loss = loss + batch_acc
                acc = acc + batch_loss
                # sequence backward
                dz = self.__backward(y_hat, y_batch)
                # sequence update data
                self.__updateData()
            # loss, acc, val_loss, val_acc
            loss = loss / batch_iteration
            acc = acc / batch_iteration
            y_hat = self.__forward(x_test.copy())
            val_loss, val_acc = self.__loss(y_hat, y_test)
            # save history
            self.__history["loss"].append(loss)
            self.__history["acc"].append(acc)
            self.__history["val_loss"].append(val_loss)
            self.__history["val_acc"].append(val_acc)
            # save best data
            self.__updateBest(loss, acc, val_loss, val_acc)
            # export result
            print(
                "Epoch {: 5d}/{}\t- loss: {:.4f} - acc: {:.4f} - val_loss: {:.4f} - val_acc: {:.4f}".format(
                    epoch, epochs, loss, acc, val_loss, val_acc
                )
            )
        print("=" * 60)
        print("Best:\n- val_loss: {:.4f}\n- val_acc: {:.4f}".format(
            self.__history_best["val_loss"],
            self.__history_best["val_acc"],
        ))
        print("-" * 60)

    def save(self, model_dir: str, labels: dict = {}):
        if os.path.exists(model_dir):
            os.remove(model_dir)
        f = h5py.File(model_dir, "w")
        # imgsz
        grp_imgsz = f.create_group("imgsz")
        for i, dim in enumerate(self.__imgsz):
            grp_imgsz[str(i)] = str(dim)
        # layers
        for layer in self.__layers:
            grp_layer = f.create_group(f"layers/{layer.id}")
            layer.save(grp_layer)
        # labels
        grp_labels = f.create_group("labels")
        for k, v in labels.items():
            grp_labels[str(k)] = v
        f.close()
        print(f"Model saved: {model_dir}")

    def load(self, model_dir: str):
        f = h5py.File(model_dir, "r")
        # imgsz
        imgsz = []
        grp = f["imgsz"]
        for k in grp.keys():
            imgsz.append(int(grp[k][()].decode('utf-8')))
        self.__imgsz = tuple(imgsz)
        # layers
        grp = f["layers"]
        for layer_id in grp:
            layer_hdf5 = grp[layer_id]
            layer = self.__getLayer(layer_hdf5)
            self.__layers.append(layer)
        # labels
        self.__labels = {}
        grp = f["labels"]
        for k in grp.keys():
            self.__labels[int(k)] = grp[k][()].decode('utf-8')
        f.close()
        print(f"Model loaded: {model_dir}")

    @property
    def labels(self):
        return self.__labels

    def __call__(self, data: np.ndarray):
        data = np.resize(data, self.__imgsz)
        y_hat = self.__forward(np.expand_dims(data, axis=0))[0]
        result = self.labels[np.argmax(y_hat)]
        scores = {}
        for k in self.labels.keys():
            scores[self.labels[k]] = "{:.3f}".format(y_hat[k])
        return result, scores

    def __forward(self, x: np.ndarray):
        for layer in self.__layers:
            x = layer.forward(x)
        return x

    def __backward(self, y_hat: np.ndarray, y: np.ndarray):
        dz = self.__layers[-1].activation.backward(y_hat, y)
        for layer in self.__layers[::-1]:
            dz = layer.backward(dz)
        return dz

    def __updateData(self):
        for layer in self.__layers:
            data = layer.data
            grad = layer.grad
            layer.data = self.__optimizer(data, grad)

    def __updateBest(self, loss, acc, val_loss, val_acc):
        if val_acc > self.__history_best["val_acc"]:
            self.__history_best = {
                "loss": loss,
                "acc": acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            for layer in self.__layers:
                layer.best = layer.data

    def __getLayer(self, layer_hdf5: h5py._hl.group.Group):
        # layer's name, args, data
        name = layer_hdf5.attrs["name"]
        args = layer_hdf5["args"]
        data = layer_hdf5["data"]
        layer = None
        # get layer instance
        if name == "Flatten":
            layer = layers.Flatten()
        if name == "Dense":
            units = int(args["units"][()])
            activation = str(args["activation"][()].decode('utf-8'))
            layer = layers.Dense(units=units, activation=activation)
        if layer:
            layer.load(data)
            return layer
        raise KeyError(
            f"No layer : `{name}`"
        )


def load(model_dir: str):
    model = Sequential()
    model.load(model_dir)
    return model
