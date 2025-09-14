






















import cv2



from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.axes

from numpy import argmax as npa
from numpy import array as npaa


import matplotlib.pyplot as plt

import  tensorflow
from tensorflow.python.keras import models


(training_images, training_labels), (testing_images, testing_labels) = keras.datasets.cifar10.load_data()


training_images, testing_images = training_images / 255, testing_images / 255

class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print('test')
# for i in range(10):
#     plt.subplot()
#     plt.yticks([])
#     plt.xticks([])
#     plt.xlabel(class_names[training_labels[i][0]])
#     plt.imshow(training_images[i], cmap=plty.cm.binary)
#     plt.xlabel(class_names[training_labels[i][0]])
#
#
# training_images = training_images[:10000]
# training_labels= training_labels[:10000]
# testing_images = testing_images[:100]
# testing_labels = testing_labels[:100]
#
#
# model = models.Sequential()
# model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(32,32,3)))
# model.add(layers.MaxPooling2D(2,2))
# model.add(layers.Conv2D(64,(3,3),activation='relu'))
# model.add(layers.MaxPooling2D(2,2))
# model.add(layers.Conv2D(64,(3,3),activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64,activation='relu'))
# model.add(layers.Dense(10,activation='softmax'))
#
#
#
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# model.fit(training_images,training_labels, epochs=10,validation_data=(testing_images,testing_labels))

# loss, accuracy = model.evaluate(testing_images, testing_labels)
#
# print(f'loss {loss}')
#
# print(f'accuracy {accuracy}')



models.load_model('image-classifier.model')

model =models.load_model('image-classifier.model')



img = cv2.imread('image-classifier.model/f354.png')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(npaa([img]) / 255)

index = npa(prediction)

print( f'prediction is {class_names[index]}')

plt.show()