import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# XOR dataset
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])
Y = np.array([[0],[1],[1],[0]])
# Build Model
model = Sequential()
model.add(Dense(8, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X, Y, epochs=1000, verbose=0)
prediction = model.predict(X)
print("Predictions")
for i in range(len(X)):
    print(X[i], "=", round(prediction[i][0]))
