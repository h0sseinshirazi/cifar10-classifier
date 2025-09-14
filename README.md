# CIFAR-10 image classifier

A convolutional classifier over the ten CIFAR-10 classes (plane, car, bird, cat,
deer, dog, frog, horse, ship, truck), in TensorFlow/Keras with OpenCV and
matplotlib. The training set is deliberately cut to 10,000 images to stay
trainable on a laptop GPU.

Two working copies survived, saved two hours apart on 2025-09-14 and recovered
from `PycharmProjects/pythonProject8` and `PycharmProjects/image ai model`.
They are committed in file-mtime order, so the diff between them is real but the
*direction* is only as trustworthy as NTFS timestamps: the copy carrying
`load_model` (inference) is the earlier of the two.

Originally shipped alongside 695 MB of packages installed with
`pip install --target .`; only the source is kept here.
