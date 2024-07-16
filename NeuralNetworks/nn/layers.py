import h5py
from itertools import count
import numpy as np
from NeuralNetworks.nn import activations


class Layer(object):
    ids = count(1)
    weight_scale = 1e-1
    cache_shape = (1,)

    def __init__(self, **kwargs) -> None:
        # args
        self.id = next(Layer.ids)
        self.kwargs = kwargs
        # input shape
        try:
            self.input_shape = self.kwargs["input_shape"]
        except:
            self.input_shape = Layer.cache_shape
        # set summary
        self.setSummary()
        # set params
        self.setParams()
        # activation
        self.getActivation()

    @property
    def summary(self):
        return {
            "layer": self.__class__.__name__,
            "output_shape": (None,) + self.output_shape,
            "param": self.param,
        }

    def setParams(self):
        self.data = {}
        self.grad = {}
        self.best = {}

    def getActivation(self):
        try:
            activation = self.kwargs["activation"].lower()
            if activation == "sigmoid":
                self.activation = activations.Sigmoid()
            if activation == "relu":
                self.activation = activations.Relu()
            if activation == "softmax":
                self.activation = activations.Softmax()
            raise KeyError(
                f"No activation : `{activation}`"
            )
        except:
            pass

    def setSummary(self):
        raise NotImplementedError

    def forward(self, x: np.ndarray):
        raise NotImplementedError

    def backward(self, dz: np.ndarray):
        raise NotImplementedError

    def save(self, grp_layer: h5py._hl.group.Group):
        # name
        grp_layer.attrs["name"] = self.__class__.__name__
        # args
        args = grp_layer.create_group("args")
        for k, v in self.kwargs.items():
            args[k] = v
        # data
        layer_data = grp_layer.create_group("data")
        for key, best in self.best.items():
            layer_data.create_dataset(key, data=best)

    def load(self, data: h5py._hl.group.Group):
        pass


class Flatten(Layer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def setSummary(self):
        self.output_shape = Layer.cache_shape = (
            np.prod(np.array(self.input_shape)),)
        self.param = 0

    def forward(self, x: np.ndarray):
        m = x.shape[0]
        n = np.prod(np.array(x.shape[1:]))
        return x.reshape(m, n)

    def backward(self, dz: np.ndarray):
        m = dz.shape[0]
        return dz.reshape((m,) + self.input_shape)


class Dense(Layer):
    cache = {
        "x": {},
        "z": {},
        "g": {},
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def setSummary(self):
        self.output_shape = Layer.cache_shape = (self.kwargs["units"],)
        self.param = (self.input_shape[0]+1) * self.output_shape[0]

    def setParams(self):
        h_l_prev = self.input_shape[0]
        h_l = self.output_shape[0]
        w = np.random.randn(h_l_prev, h_l) * Layer.weight_scale
        b = np.zeros(h_l)
        self.data = {
            "w": w,
            "b": b,
        }
        self.grad = dict.fromkeys(self.data)
        self.best = dict.fromkeys(self.data)

    def forward(self, x: np.ndarray):
        """
        z = dot(x, w) + b
        a = g(z)
        """
        w = self.data["w"]
        b = self.data["b"]
        z = np.dot(x, w) + b
        a = self.activation.forward(z)
        # cache
        Dense.cache["x"][self.id] = x
        Dense.cache["z"][self.id] = z
        Dense.cache["g"][self.id] = self.activation
        return a

    def backward(self, dz: np.ndarray):
        """
        dz_prev = [dot(dz, w)] * g'(z_prev)
        dw = 1 / m * dot(x.T, dz)
        db = 1 / m * sum(dz)
        """
        x = Dense.cache["x"][self.id]
        m = x.shape[0]
        dw = np.dot(x.T, dz) / m
        db = np.sum(dz, axis=0) / m
        self.grad = {
            "w": dw,
            "b": db,
        }
        try:
            z_prev = Dense.cache["z"][self.id-1]
            g_prev = Dense.cache["g"][self.id-1]
            w = self.data["w"]
            da = np.dot(dz, w.T)
            dz_prev = da * g_prev.backward(z_prev)
            return dz_prev
        except:
            return x

    def load(self, data: h5py._hl.group.Group):
        self.data = {
            "w": np.array(data["w"]),
            "b": np.array(data["b"]),
        }
