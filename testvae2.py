# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 19:46:06 2019
Import dependencies
@author: Admin
"""
#import pydoc
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.layers import Input, Dense, Lambda, Reshape
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras import optimizers
from tensorflow.python.client import device_lib
import keras
import os
import time

import os.path
#from noisemaker import *

#%%
'''Model parameters'''
input_shape = (28,28,1)
num_classes = 10

class_batch_size = 128
class_epochs = 12

noisetype = 'noisybin'
percorruption = 0.5

'''VAE parameters'''
batch_size = 100
original_dim = 784
latent_dim = 10
intermediate_dim = 256
nb_epoch = 100
epsilon_std = 1.0
learning_rate = 1e-3
percorruption = 0.5
noisetype = 'noisybin'
opt = 'Adam'
number = '5'
beta = 1
#%%
'''Plotting tools'''
def plot(f_image):
    first_image = np.array(f_image, dtype='float32')
    pixels = first_image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()
#%%
'''Set up classifier'''
#images = Input(input_shape)
#images = Input(batch_shape=(batch_size, original_dim))
images = Input(shape = (original_dim,))
print(images)
#%%
#imagesre = Reshape((28, 28, 1))(images)
imagesre = Reshape(input_shape, input_shape=input_shape)(images)
print(imagesre)
#%%
hidden1 = Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape)(imagesre)
hidden2 = Conv2D(64, (3, 3), activation='relu')(hidden1)
hidden2 = MaxPooling2D(pool_size=(2, 2))(hidden2)
hidden2 = Dropout(0.25)(hidden2)
hidden2 = Flatten()(hidden2)
hidden3 = Dense(128, activation = 'relu')(hidden2)
hidden3 = Dropout(0.5)(hidden3)
output = Dense(num_classes, activation='softmax')(hidden3)


classmodel = Model(inputs = images, outputs = output)

#%%
'''Check model'''
print(classmodel.summary())
plot_model(classmodel, to_file = 'convnettest.png')
#%%
'''Compile'''
classmodel.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
#%%
'''Import and preprocess data'''
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255.
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.

y_train_matrix = keras.utils.to_categorical(y_train, num_classes)
y_test_matrix = keras.utils.to_categorical(y_test, num_classes)
#%%
'''Corrupt and conquer'''
from noisemaker import *
x_train_noisy = x_train
x_test_noisy = x_test

corr = -np.log(1-percorruption)
#corr = -np.log(1-0.3)


if noisetype == 'gauss':
    x_train_noisy = np.array([add_gauss_noise(1, x_t) for x_t in x_train]).astype('float32')
    x_test_noisy = np.array([add_gauss_noise(1, x_te) for x_te in x_test]).astype('float32')
elif noisetype == 'noisybin':
    x_train_noisy = np.array([noisybin(corr, x_t) for x_t in x_train]).astype('float32')
    x_test_noisy = np.array([noisybin(corr, x_te) for x_te in x_test]).astype('float32')
elif noisetype == 'verticallines':
    x_train_noisy = np.array([noisylinesvertical(corr, x_t) for x_t in x_train]).astype('float32')
    x_test_noisy = np.array([noisylinesvertical(corr, x_te) for x_te in x_test]).astype('float32')
else:
    print('No noise?')
    
#%%
'''Testplot of the corrupted images'''
plot(x_test_noisy[2])

#%%
'''TRAIN!'''
classmodel.fit(x_train_noisy, y_train_matrix,
          batch_size=batch_size,
          epochs=class_epochs,
          verbose=1,
          validation_data=(x_test_noisy, y_test_matrix))
score = classmodel.evaluate(x_test_noisy, y_test_matrix, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('Error rate (in percent):', 100*(1-score[1]))

#%%
randominteger = np.random.randint(0,10000)
ri = randominteger
print('Index: %s ' % randominteger)
first_image = x_test_noisy[ri]
#first_image = x_test[ri]

print(np.argmax(classmodel.predict([[x_test_noisy[ri]]])))
plot(first_image)
print('True Label: %s ' % y_test[ri])
#first_image = np.array(first_image, dtype='float')
#first_image = np.expand_dims(first_image, axis = 0)
#pixels = first_image.reshape((28, 28))
#plt.imshow(pixels, cmap='gray')
#plt.show()
#print('first image shape: {}'.format(first_image.shape))
##print(first_image)
#pred = int(classmodel.predict_classes(first_image))
#pred2 = np.around(classmodel.predict(first_image), decimals = 1)
#
#print('I think this is a %s ' % pred)
##print('Numbers: {}'.format(np.arange(0,10)))
##print('ProbsArray {} '.format(pred2[0]))
#realnumber = y_test_matrix[ri]
#realnumber = np.nonzero(realnumber)
#realnumber = int(realnumber[0])
#print('True number: {}'.format(realnumber))
#%%
hiddenmodel = Model(inputs = images, outputs = hidden3)
print(hidden1)

#%%
#x_n = Flatten()(images)
h = Dense(intermediate_dim, activation='relu')(images)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)
#%%
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
#%%
print(output)
#%%
print(z)
#%%
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
#decoder_reshape = Reshape((28, 28 , 1))
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)
#decoder_logits_reshaped = decoder_reshape(x_decoded_mean)
#logits =
#%%
decoder_h = Dense(intermediate_dim, activation='relu')(z)
decoder_logits = Dense(original_dim, activation='sigmoid')(decoder_h)
decoder_logits_reshaped = Reshape((28, 28 , 1))(decoder_logits)
#h_decoded = decoder_h(z)
#x_decoded_mean = decoder_mean(h_decoded)

#print(x_decoded_mean)
#print(z)
#%%
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

x_train_noisy = x_train_noisy.reshape((len(x_train_noisy), np.prod(x_train_noisy.shape[1:])))
x_test_noisy = x_test_noisy.reshape((len(x_test_noisy), np.prod(x_test_noisy.shape[1:])))
#%%
def vae_loss(x_c, x_decoded_mean):
    xent_loss = original_dim * objectives.binary_crossentropy(x_c, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + beta*kl_loss

vae = Model(inputs = images, outputs = x_decoded_mean)
#optim = 'rmsprop'
#optim = 'adam'
#learning_rate = 1e-4
optim = optimizers.Adam(lr = learning_rate)
vae.compile(optimizer= optim, loss=vae_loss)
#%%
plot_model(vae, to_file = 'vae.png')
#%%
print(x_n)
#%%
print(x_decoded_mean)
#print(decoder_logits_reshaped)
print(images)
print(x_train_noisy.shape)
#%%
vae.fit(x_train_noisy, x_train,
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        validation_data=(x_test_noisy, x_test),verbose=1)
#%%
'''För att kunna använda VAEn'''
encoder = Model(images, [z_mean, z_log_var])
z_mean_encoded, z_logvar_encoded = encoder.predict(x_test_noisy)

decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)

decoder = Model(decoder_input, _x_decoded_mean)

#%%
def generate_and_save_images():

  fig = plt.figure(figsize=(4,4))

  for i in range(16):
      randz = np.random.normal(loc = 0, scale = 1, size = (latent_dim, ))
      predictions = np.array(decoder.predict([[randz]], batch_size = 1),dtype = 'float32').reshape(28,28)
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[:, :], cmap='gray')
      plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
#  plt.savefig(number + '/Images/image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()
  
generate_and_save_images()

#%%
 = classifyer


#%%
vae_classifier = Model(inputs = images, outputs = output)
#%%
plot_model(vae_classifier, to_file = 'vaeclassifier.png')
#%%
print(vae_classifier.summary())
