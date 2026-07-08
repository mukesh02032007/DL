# 1. Import libraries
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# 2. XOR Dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[1],[1],[0]])
# 3. Create Model
model = Sequential()
# 4. Hidden Layer
model.add(Dense(8,input_dim=2,activation='relu'))
# 5. Output Layer
model.add(Dense(1,activation='sigmoid'))
# 6. Compile
model.compile(loss='binary_crossentropy',
optimizer='adam',
metrics=['accuracy'])
# 7. Train
# Train and save history
history = model.fit(X, Y, epochs=100, verbose=0)
# Plot Cost (Loss) vs Iteration (Epoch)
plt.plot(history.history['loss'])
plt.title('Cost per Iteration')
plt.xlabel('Epoch')
plt.ylabel('Cost (Loss)')
plt.grid(True)
plt.show()
# 8. Predict
print(model.predict(X))
