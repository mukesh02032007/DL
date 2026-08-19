import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Flatten, Dense, Dropout, Rescaling, 
    RandomFlip, RandomRotation, RandomTranslation, RandomZoom
)

# %matplotlib inline

# 1. Helper Functions
def load_images_from_path(path, label):
    images = []
    labels = []
    if not os.path.exists(path):
        print(f"Warning: Path '{path}' does not exist.")
        return images, labels
        
    for file in os.listdir(path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Target size fixed to (224, 224)
            img_path = os.path.join(path, file)
            img = image.load_img(img_path, target_size=(224, 224))
            images.append(image.img_to_array(img))
            labels.append(label)
    return images, labels

def show_images(images, max_images=8):
    num_to_show = min(len(images), max_images)
    if num_to_show == 0:
        return
    fig, axes = plt.subplots(1, num_to_show, figsize=(15, 3), subplot_kw={'xticks': [], 'yticks': []})
    if num_to_show == 1:
        axes = [axes]
    for i in range(num_to_show):
        axes[i].imshow(images[i] / 255.)
    plt.show()

# 2. Dataset Loading
x_train, y_train = [], []
x_test, y_test = [], []

# Load Arctic Fox training data (label 0)
images, labels = load_images_from_path('arctic-wildlife/train/arctic_fox', 0)
show_images(images)
x_train += images
y_train += labels

# Load Walrus training data (label 2)
images, labels = load_images_from_path('arctic-wildlife/train/walrus', 2)
show_images(images)
x_train += images
y_train += labels

# Load Polar Bear test data (label 1)
images, labels = load_images_from_path('arctic-wildlife/test/polar_bear', 1)
show_images(images)
x_test += images
y_test += labels

# Convert to NumPy Arrays & Preprocess
x_train = preprocess_input(np.array(x_train))
x_test = preprocess_input(np.array(x_test))

# One-Hot Encoding
y_train_encoded = to_categorical(y_train, num_classes=3)
y_test_encoded = to_categorical(y_test, num_classes=3)

# 3. Model Architecture
base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

model = Sequential([
    Rescaling(1./255),
    RandomFlip(mode='horizontal'),
    RandomTranslation(0.2, 0.2),
    RandomRotation(0.2),
    RandomZoom(0.2),
    base_model,
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Model Training
hist = model.fit(
    x_train,
    y_train_encoded,
    validation_data=(x_test, y_test_encoded),
    batch_size=10,
    epochs=25
)

# 5. Visualizing Accuracy
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, '-', label='Training Accuracy')
plt.plot(epochs, val_acc, ':', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# 6. Evaluation & Confusion Matrix
y_predicted = model.predict(x_test)
mat = confusion_matrix(y_test_encoded.argmax(axis=1), y_predicted.argmax(axis=1))
class_labels = ['arctic fox', 'polar bear', 'walrus']

sns.set()
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted label')
plt.ylabel('Actual label')
plt.show()

# 7. Sample Predictions
# Test Arctic Fox Image (Fixed file extension .jpg)
sample_fox_path = 'arctic-wildlife/samples/arctic_fox/arctic_fox_140.jpg'
if os.path.exists(sample_fox_path):
    img = image.load_img(sample_fox_path, target_size=(224, 224))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    img_arr = image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = preprocess_input(img_arr)

    predictions = model.predict(img_arr)
    for i, label in enumerate(class_labels):
        print(f'{label}: {predictions[0][i]}')

# Test Walrus Image
sample_walrus_path = 'arctic-wildlife/samples/walrus/walrus_143.png'
if os.path.exists(sample_walrus_path):
    img = image.load_img(sample_walrus_path, target_size=(224, 224))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    img_arr = image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = preprocess_input(img_arr)

    predictions = model.predict(img_arr)
    for i, label in enumerate(class_labels):
        print(f'{label}: {predictions[0][i]}')