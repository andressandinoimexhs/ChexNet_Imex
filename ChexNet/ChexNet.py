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

import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential,datasets, layers, models
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,LearningRateScheduler
from tensorflow.keras.optimizers import Adam

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#%%
#
#./data/datainfo.csv


dataframe = pd.read_csv("/home/usuario/Documentos/GitHub/DL_Project/ChexNet/data/datainfo.csv") 

#%%

#classes =  dataframe.columns[1:].values.tolist()

classes = ['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14']

batch_size = 16
color_mode = 'rgb'  # "grayscale", "rgb", "rgba"
img_directory_path = "/home/usuario/Descargas/images/"

target_size = (224, 224)

validation_split=0.3

train_datagen = ImageDataGenerator(rescale=1./255,validation_split=validation_split)
valid_datagen = ImageDataGenerator(rescale=1./255,validation_split=validation_split)


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
    seed=1,
    validate_filenames=True,
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
    seed=1,
    validate_filenames=True,
    )

train_steps=train_set.samples//train_set.batch_size
valid_steps=train_set.samples//train_set.batch_size

#%%

def createmodel():

    model = Sequential()
    
    input_shape=(224,224,3)
    
    model.add(tf.keras.applications.DenseNet121(include_top=False,
                                              weights=None,
                                              #weights='imagenet',
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
	initial_lrate = 1e-3
	drop = 0.1
	epochs_drop = 50
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

optimizer = Adam(1e-3) # 1e-5

BinaryCrossEnt=tf.keras.metrics.BinaryCrossentropy()

model.compile(optimizer=optimizer, 
              loss = 'binary_crossentropy', 
              metrics=[BinaryCrossEnt])


checkpoint_path = '/home/usuario/Descargas/modelo.h5'
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

aa=train_set[0][0]
bb=train_set[0][1]
plt.imshow(jj[0,:,:,:])
truelabel=bb[0]
jj=aa[:1,:,:,:]
ll=model.predict(jj)
print(ll)


