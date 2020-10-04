# https://medium.com/@randerson112358/build-your-own-artificial-neural-network-using-python-f37d16be06bf

from keras.datasets import mnist
from keras import models
from keras import layers
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
import numpy as np
from sklearn import model_selection
import pandas as pd
import sklearn
from sklearn import preprocessing


# Wczytanie danych z pliku music7
data = pd.read_csv("Data/music7.data")

# Przewidywana kolumna
predict = "Genre"


# Dwa zbiory, input/output
X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])

# Process the feature data set to contain values between 0 and 1
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

# Zmiana gatunkow (string) na liczby (0-9)
le = preprocessing.LabelEncoder()
Y = le.fit_transform(Y)

# Podzial na zbiory treningowe i testowe # 80% training and 20% testing data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, random_state=4)

input_shape = 30

model = Sequential([
    Dense(1024, activation='relu', input_shape=( input_shape ,)),
    Dense(256, activation='relu'),
    #Dense(128, activation='relu'),
    #Dense(64, activation='relu'),
    #Dense(32, activation='relu'),
    Dropout(.1, input_shape=(1,)),
    Dense(10, activation='softmax')
])


model.compile(optimizer='Adam',
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
hist = model.fit(x_train, y_train,
          batch_size=64, epochs=2000,  verbose =2,  callbacks=callback, validation_split=0.2)


print("Done, accuracy: ")
print(model.evaluate(x_test, y_test, batch_size=64))


# zapisuje wytrenowany model
#model.save("music7/2/model_RMSprop.h5")
#print("Saved model to disk")

