import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from keras import ops, layers, Model
import keras 
import tensorflow as tf

class downS(layers.Layer):
    def __init__(self, unit = 32):
        super().__init__()

        self.c1 = layers.Conv2D(unit, (3, 3), padding="same", activation="relu")
        self.c2 = layers.Conv2D(unit, (3, 3), padding="same", activation="relu")
        
        self.c3 = layers.Conv2D(unit, (3, 3), padding="same", activation="relu")
        self.nomal = layers.Normalization()
        self.pool = layers.MaxPool2D()

    def call(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.nomal(x)
        pool = self.pool(x)
        pool = self.c3(pool)
        return pool, x
    

class sk(layers.Layer):
    def __init__(self, unit = 32):
        super().__init__()

        self.c1 = layers.Conv2D(unit, (3, 3), padding="same", activation="relu")
        self.c2 = layers.Conv2D(unit, (3, 3), padding="same", activation="relu")
        
    def call(self, x):
        x = self.c1(x)
        x = self.c2(x)
        return x



class upS(layers.Layer):
    def __init__(self, unit = 32):
        super().__init__()

        self.c1 = layers.Conv2D(unit, (3, 3), padding="same", activation="relu")
        self.c2 = layers.Conv2D(unit, (3, 3), padding="same", activation="relu")
        
        self.c3 = layers.Conv2D(unit, (3, 3), padding="same", activation="relu")
        self.nomal = layers.Normalization()
        self.pool = layers.Conv2DTranspose(unit, (2, 2), (2, 2), padding="same", activation="relu")

    def call(self, x, skip):
        x = self.c1(x)
        x = self.pool(x)
        x = ops.concatenate([skip, x], -1)
        x = self.c2(x)
        x = self.nomal(x)
        x = self.c3(x)
        return x


l = 5
up = [upS(1*32) for i in range(1, l)]
sks = [sk(1*32) for i in range(1, l)]
down = [downS(i*32) for i in range(1, l)]


inp = keras.Input((224, 224, 3))
x = layers.BatchNormalization()(inp)
downs = []
for i in down:
    x, _ = i(x)
    downs.append(_)

x = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(x)
for index, l in enumerate(list(zip(up, downs))[::-1]):
    up, down_ = l
    print(x, "--")
    print(down_, "++")
    # exit()
    x = up(x, sks[::-1][index](down_))

x = layers.Conv2D(64, (2, 2), padding="same")(x)
x = layers.Conv2D(32, (2, 2), padding="same")(x)
x = layers.Conv2D(1, (2, 2), padding="same")(x)

model_ = Model([inp], [x])
model_.summary()
keras.utils.plot_model(model_, to_file='./pre_model.png')