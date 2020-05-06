# -*- coding: utf-8 -*-
"""
Created on Wed May  6 14:46:42 2020

@author: Weijian HU
"""

# coding: utf-8

import numpy as np
import pandas as pd
import os
 
df = pd.read_csv('000905.csv') 
df.columns=["close","high","low","open","volumn","amout"]
 
data = df[["close","high","low","open","volumn","amout"]] 

train = np.zeros((len(data)-5, 6, 6))
for i in range(len(data)-5):
    train[i] = data[i:i+6]
 
import keras
from keras import layers
import numpy as np
 
latent_dim = 16
height = 6
width = 6
channels = 1
 
generator_input = keras.Input(shape=(latent_dim,))
x = layers.Dense(128 * 6 * 6)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((6, 6, 128))(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2DTranspose(256, 4, strides=1, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)

generator = keras.models.Model(generator_input, x)
generator.summary()
 

discriminator_input = layers.Input(shape=(height, width, channels))
x = layers.Conv2D(128, 1)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 1, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 1, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 1, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)
x = layers.Dropout(0,4)(x)
x = layers.Dense(1, activation='sigmoid')(x)
 
discriminator = keras.models.Model(discriminator_input, x)
discriminator.summary()
 
discriminator_optimizer = keras.optimizers.RMSprop(lr=8e-4, clipvalue=1.0, decay=1e-8)
discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')
 
discriminator.trainable = False
 
gan_input = keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)
 
gan_optimizer = keras.optimizers.RMSprop(lr=4e-4, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

import os
from keras.preprocessing import image

iterations = 10000
batch_size = 10
save_dir = '/home/GDdata'
 
start = 0 
for step in range(iterations):
    print('start iter %s' %step)

    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim)) 
    generated_images = generator.predict(random_latent_vectors)
   
    stop = start + batch_size
    real_images = train[start: stop].reshape([stop - start, 6, 6, 1])
   
    combined_images = np.concatenate([generated_images, real_images])
    
    labels = np.concatenate([np.ones((batch_size, 1)),
                            np.zeros((batch_size, 1))])

    labels += 0.05 * np.random.random(labels.shape)

    d_loss = discriminator.train_on_batch(combined_images, labels)

    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

    misleading_targets = np.zeros((batch_size, 1))

    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)
    
    start += batch_size
    if start > len(train) - batch_size:
        start = 0
    if step % 100 == 0:
        gan.save_weights('gan.h5')

        print('discriminator loss at step %s: %s' % (step, d_loss))
        print('adversarial loss at step %s: %s' % (step, a_loss))

        img = image.array_to_img(generated_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'generated' + str(step) + '.png'))

        img = image.array_to_img(real_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'real' + str(step) + '.png')) 
 
import matplotlib.pyplot as plt

random_latent_vectors = np.random.normal(size=(10, latent_dim))

generated_images = generator.predict(random_latent_vectors)
 
for i in range(generated_images.shape[0]):
    img = image.array_to_img(generated_images[i] * 255., scale=False)
    plt.figure()
    plt.imshow(img)
    
plt.show()
