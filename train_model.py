import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

data = []
labels = []

dataset_path = "dataset/leapGestRecog"

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)

    if os.path.isdir(person_path):
        for gesture in os.listdir(person_path):
            gesture_path = os.path.join(person_path, gesture)

            for img_name in os.listdir(gesture_path):
                img_path = os.path.join(gesture_path, img_name)

                img = cv2.imread(img_path)
                img = cv2.resize(img, (64, 64))

                data.append(img)

                label = int(gesture[:2])
                labels.append(label)

data = np.array(data) / 255.0
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(11, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=25)

loss, accuracy = model.evaluate(X_test, y_test)

print("Accuracy:", accuracy)

model.save("model/gesture_model.h5")

print("Model Saved Successfully")