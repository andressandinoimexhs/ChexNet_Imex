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


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential,datasets, layers, models
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,LearningRateScheduler
from tensorflow.keras.optimizers import Adam

#%%
path2='C:/Users/Andres/Documents/ProyectoInvestigacion/DL_Projects/ChexNet/data/'
dataframe = pd.read_csv(path2 + "datainfo.csv") 


"""
    A1: "Atelectasis"
    A2: "Cardiomegaly"
    A3: "Effusion",
    A4: "Infiltration"
    A5: "Mass"
    A6: "Nodule"
    A7: "Pneumonia"
    A8: "Pneumothorax"
    A9: "Consolidation"
    A10: "Edema"
    A11: "Emphysema"
    A12: "Fibrosis"
    A13: "Pleural_Thickening"
    A14: "Hernia"
    
"""

#%%

classes = ['A1','A2','A3','A4','A5',
           'A6','A7','A8','A9','A10',
           'A11','A12','A13','A14']

batch_size = 4
color_mode = 'rgb'  # "grayscale", "rgb", "rgba"
img_directory_path = "/Users/Andres/Desktop/images/"
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
    
    model.add(tf.keras.applications.DenseNet121(include_top=True,
                                              #weights=None,
                                              weights='C:/Users/Andres/Desktop/brucechou1983_CheXNet_Keras_0.3.0_weights.h5',
                                              input_tensor=None,
                                              input_shape=(224, 224, 3),
                                              pooling='max',
                                              classes=14,))
    
    #model.add(Dense(14,activation='sigmoid'))
    model.summary()
    return model

model = createmodel()

#%%

mdl = Sequential()
mdl2 =  Sequential()
# kk = Sequential()

kk=model.layers[0]
#kk.add(Dense(14,activation='softmax',name='Output'))
x=kk.layers[0]


x=kk.layers[1](x)

for i in range(0,100): 
    x=kk.layers[i](x)

#mdl.summary()

#mdl.load_weights('C:/Users/Andres/Desktop/brucechou1983_CheXNet_Keras_0.3.0_weights.h5')
#%%
mdl = model.layers[0]
numlayers=len(mdl.layers)
SplitModel=Model(inputs=mdl.input, outputs=mdl.layers[numlayers-1].output)

#%%

input_img = Input(shape=(224, 224, 3),name='Input')
out_SplitModel = SplitModel(input_img)

output = Dense(14,activation='softmax',name='Output')(out_SplitModel)

model2 = Model(input_img, output)

model2.summary()

#%%
#new_model = load_model('C:/Users/Andres/Desktop/brucechou1983_CheXNet_Keras_0.3.0_weights.h5')
kk.load_weights('C:/Users/Andres/Desktop/brucechou1983_CheXNet_Keras_0.3.0_weights.h5')

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
              metrics=['accuracy',BinaryCrossEnt])


checkpoint_path = 'C:/Users/Andres/Desktop/modelo.h5'
lr = LearningRateScheduler(step_decay)
es = EarlyStopping(patience=100,mode='min', verbose=1)
mc = ModelCheckpoint(checkpoint_path, 
                     monitor='val_loss', 
                     verbose=1 , 
                     save_best_only=True, 
                     mode='min')

history = model.fit(train_set,steps_per_epoch=3,
                                   validation_data=valid_set,
                                   validation_steps=3,
                                   epochs=50,verbose=1,callbacks=[es,mc,lr])

#%%


aa=train_set[0][0]
bb=train_set[0][1]
truelabel=bb[0]
jj=aa[:1,:,:,:]
ll=model.predict(jj)
print(ll)


