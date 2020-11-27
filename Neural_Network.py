# https://medium.com/@randerson112358/build-your-own-artificial-neural-network-using-python-f37d16be06bf

from keras.datasets import mnist
from keras import models
from keras import layers
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from keras.utils import to_categorical
import numpy as np
from sklearn import model_selection
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from AudioFunctions import labeling
from sklearn.decomposition import PCA

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


'''
pca_model = PCA(n_components=25)
pca_model.fit(x_train)
x_train = pca_model.transform(x_train)
x_test = pca_model.transform(x_test)
'''


## Poczatek tworzenia sieci
input_shape = 30

model = Sequential([
    #input layer
    Dense(1024, activation='relu', input_shape=( input_shape ,)),
    Dropout(.1, input_shape=(1,)),

    # 1st layer
    Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dropout(.1, input_shape=(1,)),

    # output layer
    Dense(10, activation='softmax')
])


model.compile(optimizer='Adam',
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
hist = model.fit(x_train, y_train,
          batch_size=8, epochs=200,  verbose =2, callbacks= callback,  validation_split=0.2)


print("Done, accuracy: ")
print(model.evaluate(x_test, y_test, batch_size=8))

y_pred = model.predict(x_test)
y_pred = labeling(y_pred)
cm = confusion_matrix(y_test, le.fit_transform(y_pred))

model.summary()
print(cm)

# zapisuje wytrenowany model
#model.save("music7/2/model_RMSprop.h5")
#print("Saved model to disk")

