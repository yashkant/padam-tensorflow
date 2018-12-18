from __future__ import absolute_import, division, print_function
import tensorflow as tf
import os
import sys
import numpy as np
import sklearn.metrics as metrics
from keras.datasets import cifar10
import keras.callbacks as callbacks
import keras.utils.np_utils as kutils
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras_contrib.applications import wide_resnet

from keras import backend as K
os.environ["CUDA_VISIBLE_DEVICES"]= sys.argv[1]
tf.enable_eager_execution()


batch_size = 100
nb_epoch = 100
img_rows, img_cols = 32, 32

(trainX, trainY), (testX, testY) = cifar10.load_data()

trainX = trainX.astype('float32')
trainX = (trainX - trainX.mean(axis=0)) / (trainX.std(axis=0))
testX = testX.astype('float32')
testX = (testX - testX.mean(axis=0)) / (testX.std(axis=0))

tempY = testY
trainY = kutils.to_categorical(trainY)
testY = kutils.to_categorical(testY)

generator = ImageDataGenerator(rotation_range=10,
                               width_shift_range=5./32,
                               height_shift_range=5./32,
                               horizontal_flip=True)

generator.fit(trainX, seed=0, augment=True)

init_shape = (3, 32, 32) if K.image_dim_ordering() == 'th' else (32, 32, 3)

# For WRN-16-8 put N = 2, k = 8
# For WRN-28-10 put N = 4, k = 10
# For WRN-40-4 put N = 6, k = 4

model = wide_resnet.WideResidualNetwork(weights = None)
model.summary()
plot_model(model, "WRN-28-8.png", show_shapes=False)

optimizer = tf.train.AdamOptimizer()


for e in range(epochs):
    print('Epoch', e)
    batches = 0
    batch_size = 128
    for x_batch, y_batch in datagen.flow(trainX, trainY, batch_size=batch_size):
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            loss_value = tf.losses.sparse_softmax_cross_entropy(y_batch, logits)
        
        grads = tape.gradient([loss_value, model.variables])
        optimizer.apply_gradients(zip(grads, mnist_model.variables),
                                    global_step=tf.train.get_or_create_global_step())

        batches += 1
        if batches >= len(trainX) / batch_size:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break




# loss_history = []


# for (batch, (images, labels)) in enumerate(dataset.take(400)):
#   if batch % 80 == 0:
#     print()
#   print('.', end='')
#   with tf.GradientTape() as tape:
#     logits = mnist_model(images, training=True)
#     loss_value = tf.losses.sparse_softmax_cross_entropy(labels, logits)

#   loss_history.append(loss_value.numpy())
#   grads = tape.gradient(loss_value, mnist_model.variables)
#   optimizer.apply_gradients(zip(grads, mnist_model.variables),
#                             global_step=tf.train.get_or_create_global_step())

# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
# print("Finished compiling")
# print("Allocating GPU memory")

# # model.load_weights("weights/WRN-28-8 Weights.h5")
# # print("Model loaded.")

# model.fit_generator(generator.flow(trainX, trainY, batch_size=batch_size), steps_per_epoch=len(trainX) // batch_size + 1, nb_epoch=nb_epoch,
#                    callbacks=[callbacks.ModelCheckpoint("WRN-28-8 Weights.h5", monitor="val_acc", save_best_only=True)],
#                    validation_data=(testX, testY),
#                    validation_steps=testX.shape[0] // batch_size,)

# yPreds = model.predict(testX)
# yPred = np.argmax(yPreds, axis=1)
# yPred = kutils.to_categorical(yPred)
# yTrue = testY

# accuracy = metrics.accuracy_score(yTrue, yPred) * 100
# error = 100 - accuracy
# print("Accuracy : ", accuracy)
# print("Error : ", error)
