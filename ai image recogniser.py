import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

(training_images, training_labels), (testing_images, testing_labels) = cifar10.load_data()

training_images, testing_images = training_images / 255.0, testing_images / 255.0

class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(class_names[training_labels[i][0]])
    plt.imshow(training_images[i], cmap=plt.cm.binary)

training_images = training_images[:10000]
training_labels = training_labels[:10000]
testing_images = testing_images[:100]
testing_labels = testing_labels[:100]

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f'loss {loss}')
print(f'accuracy {accuracy}')

img = cv2.imread("E:/PycharmProjects/image ai model/354.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.expand_dims(img, axis=0) / 255.0)
index = np.argmax(prediction)
print(f'prediction is {class_names[index]}')

#plt.show()
