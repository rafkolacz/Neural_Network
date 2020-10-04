from keras.models import model_from_json
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import sklearn
from keras.models import load_model
from sklearn import preprocessing
from AudioFunctions import audioAnalysis as audio
from AudioFunctions import labeling


data = pd.read_csv("Data/music7.data")
# Przewidywana kolumna
predict = "Genre"

# Dwa zbiory, input/output
X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

# Do wyciagniecia danych z konketnego utworu
# test = []
# test.append(audio("genres/rock/rock.00007.wav"))

model = load_model('model.h5')
model.summary()

min_max_scaler = preprocessing.MinMaxScaler()
x_test = min_max_scaler.fit_transform(x_test)
output = model.predict(x_test, batch_size=57)

# Testowanie modelu
prediction = labeling(output)

correct = 0
for i in range(len(y_test)):
    if y_test[i] == prediction[i]:
        correct += 1

print("Correct data:", correct, " total: ", len(y_test))
print("Accuracy: ", (correct/len(y_test))*100, "%")