import matplotlib.pyplot as plt
import os
import cv2
import keras
import numpy as np
import random
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from sklearn import model_selection
from scipy import ndimage


def get_best_shift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)

    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)

    return shiftx, shifty


def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted


def shift_according_to_center_of_mass(img):
    img = cv2.bitwise_not(img)

    # Centralize the image according to center of mass
    shiftx, shifty = get_best_shift(img)
    shifted = shift(img, shiftx, shifty)
    img = shifted

    img = cv2.bitwise_not(img)
    return img


batch_size = 64
num_classes = 9
epochs = 100

img_rows, img_cols = 32, 32

PATH = "data/Fnt"
CATEGORIES = [str(i) for i in range(1, 10)]
training_data = []
for category in CATEGORIES:
    path = os.path.join(PATH, category)
    class_num = CATEGORIES.index(category)
    for img in os.listdir(path):
        image_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        resized_array = cv2.resize(image_array, (img_rows, img_cols), interpolation=cv2.INTER_LANCZOS4)
        _, resized_array = cv2.threshold(resized_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        resized_array = shift_according_to_center_of_mass(resized_array)
        training_data.append([resized_array, class_num])
#print(len(training_data))

random.shuffle(training_data)

X = []
y = []
for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, 32, 32, 1)
X = X / 255
print(X[0])
y = np.array(y)
y = keras.utils.to_categorical(y, num_classes=9)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.10)
print(len(X_train), len(X_test))

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)

# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

model.save('models/model2.hdf5')
#model.save_weights('models/digitRecognition2.h5')