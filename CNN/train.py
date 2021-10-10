# ライブラリのインポート
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

##データセットを読み込む
(x_train, y_train), (x_test, y_test) = mnist.load_data()

## ２次元データを１次元データに変換
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

# 畳み込みニューラルネットワークに使用するためにデータを変形する
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

##モデルを定義する
model = Sequential()

##Conv②Dで2次元レイヤーを表現する
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))


# (2,2)のマックスプーリング層を追加
model.add(MaxPooling2D(pool_size=(2, 2)))

# 20%をドロップするDropOut層を追加
model.add(Dropout(0.2))

# データを１次元にする
model.add(Flatten())

# 出力：128次元の全結合層とReLU層を追加
model.add(Dense(128, activation='relu'))

# 30%をドロップするDropOut層を追加
model.add(Dropout(0.3))

#0~9の10次元の出力層とsoftmax層を追加
model.add(Dense(10, activation='softmax'))
