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
from keras.models import load_model
import numpy as np
from sklearn import model_selection
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from AudioFunctions import labeling
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def labeling(output):
    genres = []
    labels = ["Blues", "Classical", "Country", "Disco", "Hiphop", "Jazz", "Metal", "Pop", "Reggae", "Rock"]
    for song in output:
        genres.append(labels[song])
    return genres


def replace(replacement, Yte):
    for j in range(int(len(replacement) / 2)):
        for i in range(len(Yte)):
            if Yte[i] == replacement[j * 2]:
                Yte[i] = replacement[j * 2 + 1]
    return Yte

def delete(index, Yte):
    for i in range(len(index)):
        Yte = np.delete(Yte, np.s_[20 * index[i]:(20 * (index[i] + 1))], axis=0)
    return Yte

# load test and holy data
with open('Holy_Models/Xte.npy', 'rb') as f:
    Xte = np.load(f)
with open('Holy_Models/Yte.npy', 'rb') as f:
    Yte = np.load(f)

with open('Holy_Models/Xho.npy', 'rb') as f:
    Xho = np.load(f)
with open('Holy_Models/Yho.npy', 'rb') as f:
    Yho = np.load(f)

le = preprocessing.LabelEncoder()
X = Xho.copy()
Y = Yho.copy()
# load model M0
model = load_model('Models/M0.h5')

# Klasyfikacja utworu do Reszta lub Rock/Metal
M0_predicted = model.predict(X)

# Przypisanie odpowiedniej etykiety (zamiast liczb nzawy gatunkow)
# 0 - Rest 1 - Rock/Metal
predicted_M0 = []
labels = ["Bl/Cl/Ja", "Co/Di/Po/HH/Re", "Ro/Me"]
for song in M0_predicted:
    predicted_M0.append(labels[np.argmax(song)])

# tablica pomylek M0
print(predicted_M0)
Yte_M0 = replace([1, 0, 2, 1, 3, 1, 4, 1, 5, 0, 6, 2, 7, 1, 8, 1, 9, 2], Yho) #Yho lub Yte
predicted_M0 = le.fit_transform(predicted_M0)
cm = confusion_matrix(Yte_M0, predicted_M0)
print(cm)

# Podzial danych

X_M1 = np.arange(30 * np.count_nonzero(predicted_M0 == 0), dtype='f').reshape(np.count_nonzero(predicted_M0 == 0), 30)
Y_M1 =[]
X_M2 = np.arange(30 * np.count_nonzero(predicted_M0 == 2), dtype='f').reshape(np.count_nonzero(predicted_M0 == 2), 30)
Y_M2 =[]
X_M3 = np.arange(30 * np.count_nonzero(predicted_M0 == 1), dtype='f').reshape(np.count_nonzero(predicted_M0 == 1), 30)
Y_M3 =[]
count_1 = 0
count_2 = 0
count_3 = 0

for i in range(len(Yte_M0)):
    if predicted_M0[i] == 0:
        Y_M1.append(Y[i])
        X_M1[count_1] = X[i]
        count_1 = count_1 + 1
    if predicted_M0[i] == 2:
        Y_M2.append(Y[i])
        X_M2[count_2] = X[i]
        count_2 = count_2 + 1
    if predicted_M0[i] == 1:
        Y_M3.append(Y[i])
        X_M3[count_3] = X[i]
        count_3 = count_3 + 1


# load model M1
model = load_model('Models/M1.h5')

# Klasyfikacja utworu do Blues, Classical lub Jazz
M1_predicted = model.predict(X_M1)
# Przypisanie odpowiedniej etykiety (zamiast liczb nazwy gatunkow)
# 0 - Blues 1 - Classical 2 - Jazz
genres = []
new_Y = []
labels = ["Blues", "Classical", "Jazz"]
M1 = []
count = 0
for song in M1_predicted:
    M1.append(labels[np.argmax(song)])
    genres.append(labels[np.argmax(song)])
    new_Y.append(Y_M1[count])
    count = count + 1

print(M1)
print(new_Y)
# load model M2
model = load_model('Models/M2.h5')

# Klasyfikacja utworu do Blues, Classical lub Jazz
M2_predicted = model.predict(X_M2)
# Przypisanie odpowiedniej etykiety (zamiast liczb nazwy gatunkow)
# 0 - Metal 1 - Rock
labels = ["Metal", "Rock"]
M2 = []
count = 0
for song in M2_predicted:
    M2.append(labels[np.argmax(song)])
    genres.append(labels[np.argmax(song)])
    new_Y.append(Y_M2[count])
    count = count + 1



# load model M3

model = load_model('Models/M3.h5')

# Klasyfikacja utworu do Country, Pop, Reggae, Disco/Hip-Hop
M3_predicted = model.predict(X_M3)
# Przypisanie odpowiedniej etykiety (zamiast liczb nazwy gatunkow)
# 0 - Country 1 - Pop 2 - Reggae 3 - Disco/Hip-Hop
labels = ["Reggae", "Pop", "Country", "Disco/Hip-Hop"]
M3 = []
for song in M3_predicted:
    M3.append(labels[np.argmax(song)])

wat = []
for i in range(len(M3)):
    if M3[i] == "Disco/Hip-Hop":
        pass
    else:
        genres.append(M3[i])
        wat.append(M3[i])
        new_Y.append(Y_M3[i])

print(wat)
print(Y_M3)
# Podzial Disco/Hip-hop na Disco i Hip-Hop
M3 = le.fit_transform(M3)

X_M4 = np.arange(30 * np.count_nonzero(M3 == 1), dtype='f').reshape(np.count_nonzero(M3 == 1), 30)
Y_M4 =[]

count_4 = 0
for i in range(len(M3)):
    if M3[i] == 1:
        Y_M4.append(Y_M3[i])
        X_M4[count_4] = X_M3[i]
        count_4 = count_4 + 1
# load model M4

model = load_model('Models/M4.h5')

# Klasyfikacja utworu do Country, Pop, Reggae, Disco/Hip-Hop
M4_predicted = model.predict(X_M4)
# Przypisanie odpowiedniej etykiety (zamiast liczb nazwy gatunkow)
# 0 - Country 1 - Pop 2 - Reggae 3 - Disco/Hip-Hop
labels = ["Hip Hop", "Disco"]
M4 = []
for song in M4_predicted:
    M4.append(labels[np.argmax(song)])

for i in range(len(M4)):
    genres.append(M4[i])
    new_Y.append(Y_M4[i])

'''
genres = []
genres.append(M1)
genres.append(M2)
'''
#new_Y = []
#new_Y.append(Y_M1)
#new_Y.append(Y_M2)




# z M3 trzeba usuna disco/hiphop i z Ym3 to samo z tym indeksem
#for i in range(len(M33)):
    #if M33[i] == "DiHH":
     #   pass
    #else:
        #genres.append(M33[i])
        #new_Y.append(Y_M3[i])

#print(len(genres))
# polaczyc to co zostalo
print(genres)
print(le.fit_transform(genres))



print(new_Y)
cm = confusion_matrix(new_Y, le.fit_transform(genres), labels=[0,1,2,3,4,5,6,7,8,9])
print(cm)

print(genres)
print()
