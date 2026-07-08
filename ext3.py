import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
faces = fetch_lfw_people(min_faces_per_person=70,
                         resize=0.4)
X = faces.images
Y = faces.target
X = X.reshape(-1,50,37,1)
X_train,X_test,Y_train,Y_test = train_test_split(
    X,Y,test_size=0.2,random_state=42)
model = Sequential()
model.add(Conv2D(32,(3,3),
                 activation='relu',
                 input_shape=(50,37,1)))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(len(faces.target_names),
                activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train,
          Y_train,
          epochs=5)
loss,accuracy=model.evaluate(X_test,Y_test)
print("Face Recognition Accuracy:",accuracy)
