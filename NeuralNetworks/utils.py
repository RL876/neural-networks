import os
import glob
import argparse
import numpy as np


def parseArgs():
    """
    # Function:

    Merge arguments from all relative models.

    # Arguments: None

    # Returns: args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dev", "ngrok", "ops", "train"], help="mode",
                        default="ops")
    parser.add_argument("--epochs", type=int, help="epochs", default=10)
    parser.add_argument("--lr", type=float, help="learning rate", default=0.01)
    parser.add_argument("--batch", type=int, help="batch size", default=16)
    parser.add_argument("--model_save_dir", type=str, help="set model save dir for training",
                        default=os.path.join(os.getcwd(), "output"))
    parser.add_argument("--model_dir", type=str, help="set model dir for app",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "doodle.h5"))
    parser.add_argument("--datasets", type=str, help="select datasets",
                        choices=["doodle", "mnist"], default="doodle")
    args = parser.parse_args()
    return args


def getDataset(datasets: str):
    dataLoader = DataLoader()
    if datasets == "doodle":
        return dataLoader.doodle
    if datasets == "mnist":
        return dataLoader.mnist
    raise ValueError("No Datasets : {datasets}")


class DataLoader(object):
    """
    ## Class:

    This class provide default dataset process.
    Run each method named as a dataset for getting the processed dataset.
    """

    def __init__(self,
                 train_set_ratio: float = 0.8,
                 test_set_ratio: float = 0.1) -> None:
        self.train_set_ratio = train_set_ratio
        self.test_set_ratio = test_set_ratio
        self.x_train = None
        self.x_test = None
        self.x_val = None
        self.y_train = None
        self.y_test = None
        self.y_val = None
        self.label_dict = {}
        self.train_shape = None
        self.test_shape = None

    def __oneHot(self, y):
        labels = np.unique(y)
        labels.sort()
        label_dict = dict()
        for value in labels:
            key = np.where(labels == value)[0][0]
            label_dict[key] = str(value)
            y = np.where(y == value, key, y)
        y = y.astype(int)
        y = np.eye(len(np.unique(y)))[y].astype(int)

        return y, label_dict

    def __splitDataset(self, x, y):
        """
        Description : shuffle and split dataset for train, test, validation
        Shpae       : (M, ), M=numbers of dataset
        """
        row = len(y)

        # shuffle with randomize index
        randomize = np.arange(row)
        np.random.shuffle(randomize)
        x = x[randomize]
        y = y[randomize]

        # split
        train_index = int(row * self.train_set_ratio)
        test_index = train_index + int(row * self.test_set_ratio)

        x_train = x[:train_index]
        x_test = x[train_index:test_index]
        x_val = x[test_index:]
        y_train = y[:train_index]
        y_test = y[train_index:test_index]
        y_val = y[test_index:]

        return x_train, x_test, x_val, y_train, y_test, y_val

    @property
    def doodle(self):
        # load dataset
        x = np.empty((0, 28, 28, 1), int)
        y = np.empty((0), int)
        path = os.path.join("datasets", "doodle")
        filenames = glob.glob(os.path.join(path, "*.npy"))
        for filename in filenames:
            category = os.path.splitext(os.path.basename(filename))[0]
            data = np.load(filename)
            data = data.reshape(len(data), 28, 28, 1)
            data = data[:10000]
            x = np.append(x, data, axis=0)
            y = np.append(y, np.repeat(category, 10000))
        x = x / 255
        # one hot encoding
        y, label_dict = self.__oneHot(y)
        # dataset split
        x_train, x_test, x_val, y_train, y_test, y_val = \
            self.__splitDataset(x, y)
        # save to instance
        self.x_train = x_train
        self.x_test = x_test
        self.x_val = x_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val
        self.label_dict = label_dict
        self.train_shape = x_train.shape
        self.test_shape = x_test.shape
        return self

    @property
    def mnist(self):
        # load dataset
        path = os.path.join("datasets", "mnist")
        data = np.load(os.path.join(path, "mnist.npz"))
        images = np.concatenate([data["x_train"], data["x_test"]])
        labels = np.concatenate([data["y_train"], data["y_test"]])
        x = images.reshape(images.shape[0], 28, 28, 1)
        y = labels.reshape(labels.shape[0]).astype('int')
        x = x / 255
        # one hot encoding
        y, label_dict = self.__oneHot(y)
        # dataset split
        x_train, x_test, x_val, y_train, y_test, y_val = \
            self.__splitDataset(x, y)
        # save to instance
        self.x_train = x_train
        self.x_test = x_test
        self.x_val = x_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val
        self.label_dict = label_dict
        self.train_shape = x_train.shape
        self.test_shape = x_test.shape
        return self
