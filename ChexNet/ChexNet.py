"""
Created on Tue Aug 10 15:02:57 2021
@author: Andres Sandino
e-mail: asandino@unal.edu.co

"""

import tensorflow as tf
import tensorflow.keras as keras

import os
from os import listdir
import math

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential,datasets, layers, models
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,LearningRateScheduler
from tensorflow.keras.optimizers import Adam

#%%

dataframe=nq

classes =  dataframe.columns[1:].values.tolist()

batch_size = 16
color_mode = 'rgb'  # rgb 
img_directory_path = "C:/Users/Andres/Desktop/images/"
target_size = (224, 224)


train_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.4)
valid_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.4)


train_set = train_datagen.flow_from_dataframe(
    dataframe,
    directory=img_directory_path,
    x_col="filename",
    y_col=classes,
    class_mode="raw",
    target_size=target_size,
    color_mode=color_mode,
    batch_size=batch_size,
    subset='training',
    shuffle='False',
    seed=1
    )


valid_set = valid_datagen.flow_from_dataframe(
    dataframe,
    directory=img_directory_path,
    x_col="filename",
    y_col=classes,
    class_mode="raw",
    target_size=target_size,
    color_mode=color_mode,
    batch_size=batch_size,
    subset='validation',
    shuffle='False',
    seed=1
    )


train_steps=train_set.samples//train_set.batch_size
valid_steps=train_set.samples//train_set.batch_size

#%%

def createmodel():

    model = Sequential()
    
    input_shape=(224,224,3)
    
    model.add(tf.keras.applications.DenseNet121(include_top=False,
                                              #weights=None,
                                              weights='imagenet',
                                              input_tensor=None,
                                              input_shape=(224, 224, 3),
                                              pooling='max',
                                              classes=2,))
    
    model.add(Dense(14,activation='sigmoid'))
    model.summary()
    return model

model = createmodel()

#%%

def step_decay(epoch):
	initial_lrate = 1e-4
	drop = 0.1
	epochs_drop = 50
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

optimizer = Adam(1e-4) # 1e-5

BinaryCrossEnt=tf.keras.metrics.BinaryCrossentropy()

model.compile(optimizer=optimizer, 
              loss = 'binary_crossentropy', 
              metrics=['accuracy',BinaryCrossEnt])


checkpoint_path = 'C:/Users/Andres/Desktop/modelo.h5'
lr = LearningRateScheduler(step_decay)
es = EarlyStopping(patience=100,mode='min', verbose=1)
mc = ModelCheckpoint(checkpoint_path, 
                     monitor='val_loss', 
                     verbose=1 , 
                     save_best_only=True, 
                     mode='min')

history = model.fit(train_set,steps_per_epoch=train_steps,
                                   validation_data=valid_set,
                                   validation_steps=valid_steps,
                                   epochs=50,verbose=1,callbacks=[es,mc,lr])

#%%

# optimizer = Adam(1e-4) # 1e-5
# model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])

# epochs=80

# checkpoint_path = dir + 'best_model' + Exp + '.h5'
# es = EarlyStopping(patience=80,mode='min', verbose=1)
# #mc = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1 , save_best_only=True, mode='min')

# mc = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1 , save_best_only=True, mode='min', period=4)





# optimizer = Adam(1e-4)
# model.compile(optimizer=optimizer, 
#               loss = 'binary_crossentropy', 
#               metrics=['accuracy'])

# history = model.fit_generator(train_batches,steps_per_epoch=2,
#                               epochs=3,verbose=1)

#%%

aa=train_batches[0][0]
bb=train_batches[0][1]
truelabel=bb[0]
jj=aa[:1,:,:,:]
ll=model.predict(jj)
print(ll)


