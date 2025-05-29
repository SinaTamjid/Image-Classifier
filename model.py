from keras.layers import LSTM,Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from keras.models import Sequential
from keras.optimizers import Adam,SGD
from keras.datasets import cifar10
from keras.activations import relu
from keras.utils import to_categorical
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train=tf.cast(x_train,tf.float32)/255.0
x_test=tf.cast(x_test,tf.float32)/255.0


y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

model=Sequential([
               #1layer
               Conv2D(32,(3,3),activation="relu",padding="same",input_shape=(32,32,3)),
               MaxPooling2D(),
               #2 layer
               Conv2D(64,(3,3),activation="relu",padding="same"),
               MaxPooling2D(),
               #3layer
               Conv2D(128,(3,3),activation="relu",padding="same"),
               MaxPooling2D(),
               Flatten(),
               Dense(128,activation="relu"),
               Dropout(0.5),
               Dense(10,activation="softmax")

])

model.compile(
               loss="categorical_crossentropy",
               optimizer=SGD(learning_rate=0.01,momentum=0.9),
               metrics=["accuracy"]
)

history=model.fit(x_train,y_train,epochs=20,batch_size=64,verbose=1)
model.evaluate(x_test,y_test,verbose=1)
