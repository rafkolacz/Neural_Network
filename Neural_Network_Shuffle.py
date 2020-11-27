# https://medium.com/@randerson112358/build-your-own-artificial-neural-network-using-python-f37d16be06bf

from keras.datasets import mnist
from keras import models
from keras import layers
from sklearn.utils import shuffle
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Wczytanie danych z pliku music7
data = pd.read_csv("Data/music5.data")

# Podzial na podzbiory treningowy/testowy tak aby w kazdym gatunku byla taka sama liczba utworow
for i in range(10):
    arr = data.loc[(100*i):(100*i)+99]
    XYtrain, XYtest= sklearn.model_selection.train_test_split(arr, test_size=0.3, random_state=4)
    XYtest, Holy = sklearn.model_selection.train_test_split(XYtest, test_size=0.33, random_state=4)
    if i == 0:
        train = pd.DataFrame(XYtrain)
        test = pd.DataFrame(XYtest)
        holy = pd.DataFrame(Holy)
    else:
        test = pd.concat([test, XYtest])
        train = pd.concat([train, XYtrain])
        holy = pd.concat([holy, Holy])

# Przewidywana kolumna
predict = "Genre"

# Dwa zbiory, input/output
Xtr = np.array(train.drop([predict], 1))
Xte = np.array(test.drop([predict], 1))
Xho = np.array(holy.drop([predict], 1))

Ytr = np.array(train[predict])
Yte = np.array(test[predict])
Yho = np.array(holy[predict])

# Process the feature data set to contain values between 0 and 1
min_max_scaler = preprocessing.MinMaxScaler()
Xte = min_max_scaler.fit_transform(Xte)
Xtr = min_max_scaler.fit_transform(Xtr)
Xho = min_max_scaler.fit_transform(Xho)

# standardization starts ####################
X_train_stand = Xtr.copy()
X_test_stand = Xte.copy()


# apply standardization on numerical features
for i in range(31):
    # fit on training data column
    scale = StandardScaler().fit(X_train_stand[[i]])

    # transform the training data column
    X_train_stand[i] = scale.transform(X_train_stand[[i]])

    # transform the testing data column
    X_test_stand[i] = scale.transform(X_test_stand[[i]])

Xtr = X_train_stand.copy()
Xte = X_test_stand.copy()
# standardization ends ####################


# Zmiana gatunkow (string) na liczby (0-9)
le = preprocessing.LabelEncoder()
Ytr = le.fit_transform(Ytr)
Yte = le.fit_transform(Yte)
Yho = le.fit_transform(Yho)

Xtru = Xtr.copy()
Ytru = Ytr.copy()
Xteu = Xte.copy()
Yteu = Yte.copy()

'''
for i in range(700):
    if Ytr[i] == 9:
        Ytr[i] = 1
    elif Ytr[i] == 6:
        Ytr[i] = 1
    else:
        Ytr[i] = 0
for i in range(200):
    if Yte[i] == 9:
        Yte[i] = 1
    elif Yte[i] == 6:
        Yte[i] = 1
    else:
        Yte[i] = 0

'''

# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]
print(holy)
print(len(Yte))

#Xte = np.delete(Xte, np.s_[120:140], axis=0)
#Yte = np.delete(Yte, np.s_[120:140], axis=0)
#Xtr = np.delete(Xtr, np.s_[480:560], axis=0)
#Ytr = np.delete(Ytr, np.s_[480:560], axis=0)

#Xte = np.delete(Xte, np.s_[160:], axis=0)
#Yte = np.delete(Yte, np.s_[160:], axis=0)
#Xtr = np.delete(Xtr, np.s_[640:], axis=0)
#Ytr = np.delete(Ytr, np.s_[640:], axis=0)
# PCA
'''
pca_model = PCA(n_components=29)
pca_model.fit(Xtr)
x_train = pca_model.transform(Xtr)
x_test = pca_model.transform(Xte)
'''

# Usuwa kolumny (trening co 80, test co 20!)
'''
genres = 10 # ile ma byc gatunkow

Xte = np.delete(Xte, np.s_[(20*genres):], axis=0)
Yte = np.delete(Yte, np.s_[(20*genres):], axis=0)
Xtr = np.delete(Xtr, np.s_[(80*genres):], axis=0)
Ytr = np.delete(Ytr, np.s_[(80*genres):], axis=0)
'''



# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

print(len(Yte))
print(Ytr)
print(Yte)
# Shuffle
Ytr, Xtr = shuffle(Ytr, Xtr, random_state=0)
Yte, Xte = shuffle(Yte, Xte, random_state=0)

## Poczatek tworzenia sieci
input_shape = 28

model = Sequential([
    #input layer
    Dense(512, activation='relu', input_shape=( input_shape ,)),
    Dropout(.2, input_shape=(1,)),

    # 1st layer
    Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dropout(.2, input_shape=(1,)),

    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dropout(.2, input_shape=(1,)),

    Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dropout(.2, input_shape=(1,)),
    # output layer
    Dense(10, activation='softmax')
])


model.compile(optimizer='Adam',
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
hist = model.fit(Xtr, Ytr,
          batch_size=8, epochs=200,  verbose =2, callbacks= callback,  validation_split=0.2)


print("Done, accuracy: ")
print(model.evaluate(Xte, Yte, batch_size=8))

y_pred = model.predict(Xte)
y_pred = labeling(y_pred)
cm = confusion_matrix(Yte, le.fit_transform(y_pred))

#model.summary()
print(cm)


# zapisuje wytrenowany model
#model.save("RMM_model.h5")
#np.save('Xtr', Xtru)
#np.save('Ytr', Ytru)
#np.save('Xte', Xteu)
#np.save('Yte', Yteu)
#np.save('RMM_Xte', Xte)
#np.save('RMM_Yte', Yte)
#np.save('Xho', Xho)
#np.save('Yho', Yho)
print("Saved model to disk")

