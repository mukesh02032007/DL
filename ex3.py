import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from collections import Counter
# Load Dataset
faces = fetch_lfw_people(
    min_faces_per_person=100,
    resize=1.0,
    slice_=(slice(60, 188), slice(60, 188)),
    color=False   # <-- use grayscale
)
print("Classes:")
print(faces.target_names)
print("\nImage shape:")
print(faces.images.shape)
# Display Sample Images
fig, ax = plt.subplots(3, 6, figsize=(15, 8))
for i, a in enumerate(ax.flat):
    if i < len(faces.images):
        a.imshow(faces.images[i], cmap="gray")
        a.set_xlabel(faces.target_names[faces.target[i]])
    a.set_xticks([])
    a.set_yticks([])
plt.tight_layout()
plt.show()
# Class Distribution
pd.DataFrame.from_dict(
    {faces.target_names[k]: v for k, v in Counter(faces.target).items()},
    orient='index',
    columns=['Count']
).plot(kind='bar', legend=False)
plt.title("Class Distribution")
plt.ylabel("Images")
plt.show()
# Select 100 Images Per Class
mask = np.zeros(len(faces.target), dtype=bool)
for t in np.unique(faces.target):
    idx = np.where(faces.target == t)[0]
    mask[idx[:min(100, len(idx))]] = True
X = faces.images[mask].astype("float32") / 255.0
X = np.expand_dims(X, axis=-1)  # add channel dimension (grayscale → 1)
# Integer labels
y_labels = faces.target[mask]
# One-hot labels
y = to_categorical(y_labels)
# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y_labels,
    random_state=42
)
print("Training shape:", x_train.shape)
print("Testing shape :", x_test.shape)
# CNN Model
input_shape = (X.shape[1], X.shape[2], 1)
model = Sequential([
    Input(shape=input_shape),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(faces.target_names), activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()
# Train Model
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=20,
    batch_size=25,
    verbose=1
)
# Accuracy Plot
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()
# Confusion Matrix
pred = model.predict(x_test)
cm = confusion_matrix(
    np.argmax(y_test, axis=1),
    np.argmax(pred, axis=1)
)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=faces.target_names,
    yticklabels=faces.target_names
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
# Predict New Image
image_path = "george.jpg"   # Change path if needed
img = image.load_img(
    image_path,
    target_size=(X.shape[1], X.shape[2]),
    color_mode='grayscale'   # <-- match training data
)
plt.imshow(img, cmap="gray")
plt.title("Test Image")
plt.axis("off")
plt.show()
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # shape (1,128,128,1)
result = model.predict(img_array)[0]
print("\nPrediction Probabilities:\n")
for name, prob in zip(faces.target_names, result):
    print(f"{name:25s}: {prob:.4f}")
predicted_class = np.argmax(result)
print("\nPredicted Person:")
print(faces.target_names[predicted_class])
print("Confidence:")
print(f"{result[predicted_class] * 100:.2f}%")
