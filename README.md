# Neural Networks

Neural networks demo by only using numpy to construct the structure.

[Demo](https://neural-networks-riz6sqn55q-de.a.run.app)

---

# Tutorial

## Install

```bash
pip3 install --force-reinstall git+https://github.com/RL876/neural-networks.git
```

----

## Datasets Download

### Doodle

[Doodle](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap;tab=objects?prefix=&forceOnObjectsSortingFiltering=false&pli=1)

[bicycle.npy](https://storage.cloud.google.com/quickdraw_dataset/full/numpy_bitmap/bicycle.npy)

[cat.npy](https://storage.cloud.google.com/quickdraw_dataset/full/numpy_bitmap/cat.npy)

[flower.npy](https://storage.cloud.google.com/quickdraw_dataset/full/numpy_bitmap/flower.npy)

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
model.add(layers.Flatten(input_shape=input_shape))
model.add(layers.Dense(units=64, activation="Relu"))
model.add(layers.Dense(units=16, activation="Relu"))
model.add(layers.Dense(units=len(data.label_dict),
                       activation="Softmax"))
model.summary()
model.compile(optimizer=optimizers.AdaGrad(learning_rate=0.01),
              loss=losses.CategoricalCrossentropy())
model.fit(x_train,
          y_train,
          epochs=10,
          batch_size=16,
          validation_data=(x_test, y_test))
model.save(model_dir=model_dir, labels=data.label_dict)
```

----

## Infer

```python
from NeuralNetworks.nn import models

model = models.load(model_dir)
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

* SoftMax

$$\sigma(z)=\cfrac{e^{z}}{\sum{e^{z}}}=a$$

$$\sigma'(z)=Y_{hat}-z$$

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
Given \quad \mathbb{X}_ {(m,n)}&=\begin{vmatrix}
x_ {11} & x_ {12} & ... & x_ {1n}\\
x_ {21} & x_ {22} & ... & x_ {2n}\\
| & | & ... & |\\
x_ {m1} & x_ {m2} & ... & x_ {mn}
\end{vmatrix}
\\
\\
Given\quad \mathbb{Y}_ {(m,1)}&=\begin{vmatrix}
y_ {1}\\
y_ {2}\\
|\\
y_ {m}
\end{vmatrix}=\begin{vmatrix}
cat\\
dog\\
|\\
cat
\end{vmatrix}
\end{aligned}$$

----

## Onehot Encoding

$$\mathbb{Y}_ {(m,k)}=\begin{vmatrix}
y_ {11} & y_ {12} & ... & y_ {1k}\\
y_ {21} & y_ {22} & ... & y_ {2k}\\
| & | & ... & |\\
y_ {m1} & y_ {m2} & ... & y_ {mk}
\end{vmatrix}=\begin{vmatrix}
1 & 0 & ... & 0\\
0 & 1 & ... & 0\\
| & | & ... & |\\
1 & 0 & ... & 0
\end{vmatrix}$$

----

## Hidden Layer
Assume we have 'L' hidden layers neural networks, each layer neural have $h_ {l}$ units.

$$\begin{aligned}
Initial\quad \mathbb{W}^{[l]}_ {(h_ {l-1},h_ {l})}&=\begin{vmatrix}
w_ {11} & w_ {12} & ... & w_ {1h_ {l}}\\
w_ {21} & w_ {22} & ... & w_ {2h_ {l}}\\
| & | & ... & |\\
w_ {h_ {l-1}1} & w_ {h_ {l-1}2} & ... & w_ {h_ {l-1}h_ {l}}
\end{vmatrix}
\\
\\
Initial\quad \mathbb{B}^{[l]}_ {(1,h_ {l})}&=\begin{vmatrix}
b_ {1} & b_ {2} & ... & b_ {h_ {l}}
\end{vmatrix}
\end{aligned}$$

----

## Forward Propagation

$$\begin{aligned}
X^{[0]}_ {(m,n)}&=X_ {(m,n)}\\
\\
X^{[1]}_ {(m, h_ {1})}&=\sigma^{[1]}(Z^{[1]}_ {(m, h_ {1})})=\sigma^{[1]}(X^{[0]}_ {(m,n)}{\ \cdot\ }W^{[1]}_ {(n,h_ {1})}+B^{[1]}_ {(1,h_ {1})})\\
\\
X^{[2]}_ {(m, h_ {2})}&=\sigma^{[2]}(Z^{[2]}_ {(m, h_ {2})})=\sigma^{[2]}(X^{[1]}_ {(m, h_ {1})}{\ \cdot\ }W^{[2]}_ {(h_ {1},h_ {2})}+B^{[2]}_ {(1,h_ {2})})\\
\\
&\bullet\\
&\bullet\\
&\bullet\\
\\
X^{[l]}_ {(m, h_ {l})}&=\sigma^{[l]}(Z^{[l]}_ {(m, h_ {l})})=\sigma^{[l]}(X^{[l-1]}_ {(m, h_ {l-1})}{\ \cdot\ }W^{[l]}_ {(h_ {l-1},h_ {l})}+B^{[l]}_ {(1,h_ {l})})\\
\\
\hat{Y}_ {(m,k)}&=g(Z^{[o]}_ {(m,k)})=g^{[o]}(X^{[l]}_ {(m, h_ {l})}{\ \cdot\ }W^{[o]}_ {(h_ {l},k)}+B^{[o]}_ {(1,k)})\\
\end{aligned}$$

----

## Backward Propagation

### Output Layer (Softmax)

$$\begin{cases}
\cfrac{∂C}{∂Z}^{[o]}_ {(m,k)}&=\hat{Y}_ {(m,k)}-Y_ {(m,k)}\\
\\
\cfrac{∂C}{∂W}^{[o]}_ {(h_ {l},k)}&=\cfrac{1}{m}X^{[l]T}_ {(h_ {l},m)}{\ \cdot\ }\cfrac{∂C}{∂Z}^{[o]}_ {(m,k)}\\
\\
\cfrac{∂C}{∂B}^{[o]}_ {(1,k)}&=\cfrac{1}{m}\sum{(\cfrac{∂C}{∂Z}^{[o]}_ {(m,k)})}\\
\end{cases}$$

### Hidden Layer

$$\begin{cases}
\cfrac{∂C}{∂Z}^{[l]}_ {(m,h_ {l})}&=[\cfrac{∂C}{∂Z}^{[o]}_ {(m,k)}{\ \cdot\ }W^{[o]T}_ {(k,h_ {l})}]\times[\sigma'^{[l]}(Z^{[l]}_ {(m,h_ {l})})]\\
\\
\cfrac{∂C}{∂W}^{[l]}_ {(h_ {l-1},h_ {l})}&=\cfrac{1}{m}X^{[l]T}_ {(h_ {l-1},m)}{\ \cdot\ }\cfrac{∂C}{∂Z}^{[o]}_ {(m,h_ {l})}\\
\\
\cfrac{∂C}{∂B}^{[o]}_ {(1,h_ {l})}&=\cfrac{1}{m}\sum{(\cfrac{∂C}{∂Z}^{[l]}_ {(m,h_ {l})})}\\
\end{cases}$$

$\quad$

$$\begin{cases}
\cfrac{∂C}{∂Z}^{[l-1]}_ {(m,h_ {l-1})}&=[\cfrac{∂C}{∂Z}^{[l]}_ {(m,h_ {l})}{\ \cdot\ }W^{[l]T}_ {(h_ {l},h_ {l-1})}]\times[\sigma'^{[l-1]}(Z^{[l-1]}_ {(m,h_ {l-1})})]\\
\\
\cfrac{∂C}{∂W}^{[l-1]}_ {(h_ {l-2},h_ {l-1})}&=\cfrac{1}{m}X^{[l-2]T}_ {(h_ {l-2},m)}{\ \cdot\ }\cfrac{∂C}{∂Z}^{[l]}_ {(m,h_ {l-1})}\\
\\
\cfrac{∂C}{∂B}^{[l-1]}_ {(1,h_ {l-1})}&=\cfrac{1}{m}\sum{(\cfrac{∂C}{∂Z}^{[l-1]}_ {(m,h_ {l-1})})}\\
\end{cases}$$

$$\begin{aligned}
\\
&\quad \quad \quad \quad \quad \quad \bullet\\
&\quad \quad \quad \quad \quad \quad \bullet\\
&\quad \quad \quad \quad \quad \quad \bullet\\
\\
\end{aligned}$$

$$\begin{cases}
\cfrac{∂C}{∂Z}^{[2]}_ {(m,h_ {2})}&=[\cfrac{∂C}{∂Z}^{[3]}_ {(m,h_ {3})}{\ \cdot\ }W^{[3]T}_ {(h_ {3},h_ {2})}]\times[\sigma'^{[2]}(Z^{[2]}_ {(m,h_ {2})})]\\
\\
\cfrac{∂C}{∂W}^{[2]}_ {(h_ {1},h_ {2})}&=\cfrac{1}{m}X^{[1]T}_ {(h_ {1},m)}{\ \cdot\ }\cfrac{∂C}{∂Z}^{[2]}_ {(m,h_ {2})}\\
\\
\cfrac{∂C}{∂B}^{[2]}_ {(1,h_ {2})}&=\cfrac{1}{m}\sum{(\cfrac{∂C}{∂Z}^{[2]}_ {(m,h_ {2})})}\\
\end{cases}$$

### Input Layer

$$\begin{cases}
\cfrac{∂C}{∂Z}^{[1]}_ {(m,h_ {1})}&=[\cfrac{∂C}{∂Z}^{[2]}_ {(m,h_ {2})}{\ \cdot\ }W^{[2]T}_ {(h_ {2},h_ {1})}]\times[\sigma'^{[1]}(Z^{[1]}_ {(m,h_ {1})})]\\
\\
\cfrac{∂C}{∂W}^{[1]}_ {(n,h_ {1})}&=\cfrac{1}{m}X^{[0]T}_ {(n,m)}{\ \cdot\ }\cfrac{∂C}{∂Z}^{[1]}_ {(m,h_ {1})}\\
\\
\cfrac{∂C}{∂B}^{[1]}_ {(1,h_ {1})}&=\cfrac{1}{m}\sum{(\cfrac{∂C}{∂Z}^{[1]}_ {(m,h_ {1})})}\\
\end{cases}$$

----

## Gradient Descent (Optimizers)

$$\begin{aligned}
\mathbb{W}^{[l]}_ {(h_ {l-1},h_ {l})}&=\mathbb{W}^{[l]}_ {(h_ {l-1},h_ {l})}-\alpha\cfrac{∂C}{∂W}^{[l]}_ {(h_ {l-1},h_ {l})}
\\
\\
\mathbb{B}^{[l]}_ {(1,h_ {l})}&=\mathbb{B}^{[l]}_ {(1,h_ {l})}-\alpha\cfrac{∂C}{∂B}^{[l]}_ {(1,h_ {l})}
\end{aligned}$$

---

# Reference

[Deep Dive into Math Behind Deep Networks](https://towardsdatascience.com/https-medium-com-piotr-skalski92-deep-dive-into-deep-networks-math-17660bc376ba)
