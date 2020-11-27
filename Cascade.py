import numpy as np
import tensorflow as tf
import sklearn
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from sklearn.utils import shuffle
from tensorflow.python.keras.models import load_model

def labeling(output):
    genres = []
    labels = [0, 1, 2]
    for song in output:
        genres.append(labels[np.argmax(song)])
    return genres

# load trening and test data
with open('Holy_Models/Xtr.npy', 'rb') as f:
    Xtr = np.load(f)
with open('Holy_Models/Ytr.npy', 'rb') as f:
    Ytr= np.load(f)

with open('Holy_Models/Xte.npy', 'rb') as f:
    Xte = np.load(f)
with open('Holy_Models/Yte.npy', 'rb') as f:
    Yte = np.load(f)

# delete genre from dataset

#index = []  # M0
# index = [2, 2, 2, 3, 3, 3, 3]   # M1
# index = [0, 0, 0, 0, 0, 0, 1, 1]  # M2
# index = [0,0,3,3,5]   #M3
index = [0,0,0,2,2,2,2,2]   #M4
for i in range(len(index)):

    Xte = np.delete(Xte, np.s_[20*index[i]:(20*(index[i]+1))], axis=0)
    Yte = np.delete(Yte, np.s_[20*index[i]:(20*(index[i]+1))], axis=0)
    Xtr = np.delete(Xtr, np.s_[70*index[i]:(70*(index[i]+1))], axis=0)
    Ytr = np.delete(Ytr, np.s_[70*index[i]:(70*(index[i]+1))], axis=0)
    print(index[i])

# rename ktory ktorym
#              #     #     #     #     #     #     #     #     #
#replacement = [1, 0, 2, 1, 3, 1, 4, 1, 5, 0, 6, 2, 7, 1, 8, 1, 9, 2] # M0
#replacement = [5, 2] # M1
#replacement = [9, 1, 6, 0] #M2
#replacement = [8, 0, 7, 1, 4, 3] #M3
replacement = [4,0,3,1]     #M4

for j in range(int(len(replacement)/2)):
    for i in range(len(Ytr)):
        if Ytr[i] == replacement[j*2]:
            Ytr[i] = replacement[j*2+1]

    for i in range(len(Yte)):
        if Yte[i] == replacement[j*2]:
            Yte[i] = replacement[j*2+1]

print(Yte)
print(Ytr)

Ytr, Xtr = shuffle(Ytr, Xtr, random_state=0)
Yte, Xte = shuffle(Yte, Xte, random_state=0)

le = preprocessing.LabelEncoder()

## Poczatek tworzenia sieci
input_shape = 30

model = Sequential([
    #input layer
    Dense(256, activation='relu', input_shape=( input_shape ,),kernel_regularizer=regularizers.l2(0.001)),
    Dropout(.2, input_shape=(1,)),

    # 1st layer
    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dropout(.2, input_shape=(1,)),


    # output layer
    Dense(2, activation='softmax') # ile gatunkow na output
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
print(cm)

#model.save("M4.h5")
