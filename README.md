# Neural Networks

Neural networks demo by only using numpy to construct the structure.

[Doodle Demo](https://e6vdifj1v0.execute-api.us-east-1.amazonaws.com/default/doodle)

---

# Tutorial

## Install

```bash
pip3 install --force-reinstall git+https://github.com/RL876/neural-networks.git
```

----

## Datasets Download

### Doodle

[Doodle](https://console.cloud.google.com/storage/browser/quickdraw_dataset/sketchrnn;tab=objects?prefix=&forceOnObjectsSortingFiltering=false)

[bicycle.npy](https://storage.cloud.google.com/quickdraw\_dataset/full/numpy\_bitmap/bicycle.npy)

[cat.npy](https://storage.cloud.google.com/quickdraw\_dataset/full/numpy\_bitmap/cat.npy)

[flower.npy](https://storage.cloud.google.com/quickdraw\_dataset/full/numpy\_bitmap/flower.npy)

### Mnist

[Mnist](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz)

----

## App

```bash
python3 -m NeuralNetworks.run
```
```bash
NeuralNetworks
```

----

## Train

```python
from NeuralNetworks.nn import models, layers, optimizers, losses

model = models.Sequential()
model.add(layers.Flatten(input\_shape=input\_shape))
model.add(layers.Dense(units=64, activation="Relu"))
model.add(layers.Dense(units=16, activation="Relu"))
model.add(layers.Dense(units=len(data.label\_dict),
                       activation="Softmax"))
model.summary()
model.compile(optimizer=optimizers.AdaGrad(learning\_rate=0.01),
              loss=losses.CategoricalCrossentropy())
model.fit(x\_train,
          y\_train,
          epochs=10,
          batch\_size=16,
          validation\_data=(x\_test, y\_test))
model.save(model\_dir=model\_dir, labels=data.label\_dict)
```

----

## Infer

```python
from NeuralNetworks.nn import models

model = models.load(model\_dir)
results = model(data=img)
print(results)
```

---

# Activation Function
* sigmoid function

$$\sigma(z)=\cfrac{1}{1+e^{-z}}=a$$
    
$$\sigma'(z)=\sigma(z)(1-\sigma(z))=a(1-a)$$

* tanh (second choise)
    
$$\sigma(z)=tanh(z)=\cfrac{e^{z}-e^{-z}}{e^{z}+e^{-z}}=a$$
    
$$\sigma'(z)=1-(tanh(z))^2=1-a^2$$

* ReLU (most people using for non-negative values)
    
$$\sigma(z)=max(0,z)$$

$$\sigma'(z)=\begin{cases}0,\quad if\ z<0\\
1,\quad if\ z\ge0
\end{cases}$$

* Leaky ReLU

$$\sigma(z)=max(0.01z,z)$$

$$\sigma'(z)=\begin{cases}0.01,\quad if\ z<0\\
1,\quad \quad \ if\ z\ge0
\end{cases}$$

---

# Loss Function

* MSE

$$loss=\sqrt{(\hat{Y} - Y)^2}$$

* CategoricalCrossentropy

$$loss=-(Y*log(\hat{Y}) + (1-Y)*log(1-\hat{Y}))$$

---

# Cost Function(Loss Summarize)

$$C=\cfrac{1}{m}\sum{L(\hat{Y},Y)}$$

---

# Dense(DNN)
## Dataset
Assume we have 'm' examples with 'n' parameters and 'k' categories.

$$\begin{aligned}
Given \quad \mathbb{X}\_{(m,n)}&=\begin{vmatrix}
x\_{11} & x\_{12} & ... & x\_{1n}\\
x\_{21} & x\_{22} & ... & x\_{2n}\\
| & | & ... & |\\
x\_{m1} & x\_{m2} & ... & x\_{mn}
\end{vmatrix}
\\
\\
Given\quad \mathbb{Y}\_{(m,1)}&=\begin{vmatrix}
y\_{1}\\
y\_{2}\\
|\\
y\_{m}
\end{vmatrix}=\begin{vmatrix}
cat\\
dog\\
|\\
cat
\end{vmatrix}
\end{aligned}$$

----

## Onehot Encoding

$$\mathbb{Y}\_{(m,k)}=\begin{vmatrix}
y\_{11} & y\_{12} & ... & y\_{1k}\\
y\_{21} & y\_{22} & ... & y\_{2k}\\
| & | & ... & |\\
y\_{m1} & y\_{m2} & ... & y\_{mk}
\end{vmatrix}=\begin{vmatrix}
1 & 0 & ... & 0\\
0 & 1 & ... & 0\\
| & | & ... & |\\
1 & 0 & ... & 0
\end{vmatrix}$$

----

## Hidden Layer
Assume we have 'L' hidden layers neural networks, each layer neural have $h\_{l}$ units.

$$\begin{aligned}
Initial\quad \mathbb{W}^{[l]}\_{(h\_{l-1},h\_{l})}&=\begin{vmatrix}
w\_{11} & w\_{12} & ... & w\_{1h\_{l}}\\
w\_{21} & w\_{22} & ... & w\_{2h\_{l}}\\
| & | & ... & |\\
w\_{h\_{l-1}1} & w\_{h\_{l-1}2} & ... & w\_{h\_{l-1}h\_{l}}
\end{vmatrix}
\\
\\
Initial\quad \mathbb{B}^{[l]}\_{(1,h\_{l})}&=\begin{vmatrix}
b\_{1} & b\_{2} & ... & b\_{h\_{l}}
\end{vmatrix}
\end{aligned}$$

----

## Forward Propagation

$$\begin{aligned}
X^{[0]}\_{(m,n)}&=X\_{(m,n)}\\
\\
X^{[1]}\_{(m, h\_{1})}&=\sigma^{[1]}(Z^{[1]}\_{(m, h\_{1})})=\sigma^{[1]}(X^{[0]}\_{(m,n)}{\ \cdot\ }W^{[1]}\_{(n,h\_{1})}+B^{[1]}\_{(1,h\_{1})})\\
\\
X^{[2]}\_{(m, h\_{2})}&=\sigma^{[2]}(Z^{[2]}\_{(m, h\_{2})})=\sigma^{[2]}(X^{[1]}\_{(m, h\_{1})}{\ \cdot\ }W^{[2]}\_{(h\_{1},h\_{2})}+B^{[2]}\_{(1,h\_{2})})\\
\\
&\bullet\\
&\bullet\\
&\bullet\\
\\
X^{[l]}\_{(m, h\_{l})}&=\sigma^{[l]}(Z^{[l]}\_{(m, h\_{l})})=\sigma^{[l]}(X^{[l-1]}\_{(m, h\_{l-1})}{\ \cdot\ }W^{[l]}\_{(h\_{l-1},h\_{l})}+B^{[l]}\_{(1,h\_{l})})\\
\\
\hat{Y}\_{(m,k)}&=g(Z^{[o]}\_{(m,k)})=g^{[o]}(X^{[l]}\_{(m, h\_{l})}{\ \cdot\ }W^{[o]}\_{(h\_{l},k)}+B^{[o]}\_{(1,k)})\\
\end{aligned}$$

----

## Backward Propagation

### Output Layer (Softmax)

$$\begin{cases}
\cfrac{∂C}{∂Z}^{[o]}\_{(m,k)}&=\hat{Y}\_{(m,k)}-Y\_{(m,k)}\\
\\
\cfrac{∂C}{∂W}^{[o]}\_{(h\_{l},k)}&=\cfrac{1}{m}X^{[l]T}\_{(h\_{l},m)}{\ \cdot\ }\cfrac{∂C}{∂Z}^{[o]}\_{(m,k)}\\
\\
\cfrac{∂C}{∂B}^{[o]}\_{(1,k)}&=\cfrac{1}{m}\sum{(\cfrac{∂C}{∂Z}^{[o]}\_{(m,k)})}\\
\end{cases}$$

### Hidden Layer

$$\begin{cases}
\cfrac{∂C}{∂Z}^{[l]}\_{(m,h\_{l})}&=[\cfrac{∂C}{∂Z}^{[o]}\_{(m,k)}{\ \cdot\ }W^{[o]T}\_{(k,h\_{l})}]\times[\sigma'^{[l]}(Z^{[l]}\_{(m,h\_{l})})]\\
\\
\cfrac{∂C}{∂W}^{[l]}\_{(h\_{l-1},h\_{l})}&=\cfrac{1}{m}X^{[l]T}\_{(h\_{l-1},m)}{\ \cdot\ }\cfrac{∂C}{∂Z}^{[o]}\_{(m,h\_{l})}\\
\\
\cfrac{∂C}{∂B}^{[o]}\_{(1,h\_{l})}&=\cfrac{1}{m}\sum{(\cfrac{∂C}{∂Z}^{[l]}\_{(m,h\_{l})})}\\
\end{cases}$$

$\quad$

$$\begin{cases}
\cfrac{∂C}{∂Z}^{[l-1]}\_{(m,h\_{l-1})}&=[\cfrac{∂C}{∂Z}^{[l]}\_{(m,h\_{l})}{\ \cdot\ }W^{[l]T}\_{(h\_{l},h\_{l-1})}]\times[\sigma'^{[l-1]}(Z^{[l-1]}\_{(m,h\_{l-1})})]\\
\\
\cfrac{∂C}{∂W}^{[l-1]}\_{(h\_{l-2},h\_{l-1})}&=\cfrac{1}{m}X^{[l-2]T}\_{(h\_{l-2},m)}{\ \cdot\ }\cfrac{∂C}{∂Z}^{[l]}\_{(m,h\_{l-1})}\\
\\
\cfrac{∂C}{∂B}^{[l-1]}\_{(1,h\_{l-1})}&=\cfrac{1}{m}\sum{(\cfrac{∂C}{∂Z}^{[l-1]}\_{(m,h\_{l-1})})}\\
\end{cases}$$

$$\begin{aligned}
\\
&\quad \quad \quad \quad \quad \quad \bullet\\
&\quad \quad \quad \quad \quad \quad \bullet\\
&\quad \quad \quad \quad \quad \quad \bullet\\
\\
\end{aligned}$$

$$\begin{cases}
\cfrac{∂C}{∂Z}^{[2]}\_{(m,h\_{2})}&=[\cfrac{∂C}{∂Z}^{[3]}\_{(m,h\_{3})}{\ \cdot\ }W^{[3]T}\_{(h\_{3},h\_{2})}]\times[\sigma'^{[2]}(Z^{[2]}\_{(m,h\_{2})})]\\
\\
\cfrac{∂C}{∂W}^{[2]}\_{(h\_{1},h\_{2})}&=\cfrac{1}{m}X^{[1]T}\_{(h\_{1},m)}{\ \cdot\ }\cfrac{∂C}{∂Z}^{[2]}\_{(m,h\_{2})}\\
\\
\cfrac{∂C}{∂B}^{[2]}\_{(1,h\_{2})}&=\cfrac{1}{m}\sum{(\cfrac{∂C}{∂Z}^{[2]}\_{(m,h\_{2})})}\\
\end{cases}$$

### Input Layer

$$\begin{cases}
\cfrac{∂C}{∂Z}^{[1]}\_{(m,h\_{1})}&=[\cfrac{∂C}{∂Z}^{[2]}\_{(m,h\_{2})}{\ \cdot\ }W^{[2]T}\_{(h\_{2},h\_{1})}]\times[\sigma'^{[1]}(Z^{[1]}\_{(m,h\_{1})})]\\
\\
\cfrac{∂C}{∂W}^{[1]}\_{(n,h\_{1})}&=\cfrac{1}{m}X^{[0]T}\_{(n,m)}{\ \cdot\ }\cfrac{∂C}{∂Z}^{[1]}\_{(m,h\_{1})}\\
\\
\cfrac{∂C}{∂B}^{[1]}\_{(1,h\_{1})}&=\cfrac{1}{m}\sum{(\cfrac{∂C}{∂Z}^{[1]}\_{(m,h\_{1})})}\\
\end{cases}$$

----

## Gradient Descent (Optimizers)

$$\begin{aligned}
\mathbb{W}^{[l]}\_{(h\_{l-1},h\_{l})}&=\mathbb{W}^{[l]}\_{(h\_{l-1},h\_{l})}-\alpha\cfrac{∂C}{∂W}^{[l]}\_{(h\_{l-1},h\_{l})}
\\
\\
\mathbb{B}^{[l]}\_{(1,h\_{l})}&=\mathbb{B}^{[l]}\_{(1,h\_{l})}-\alpha\cfrac{∂C}{∂B}^{[l]}\_{(1,h\_{l})}
\end{aligned}$$

---

# Reference

[Deep Dive into Math Behind Deep Networks](https://towardsdatascience.com/https-medium-com-piotr-skalski92-deep-dive-into-deep-networks-math-17660bc376ba)
