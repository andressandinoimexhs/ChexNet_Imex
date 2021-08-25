"""
Created on Tue Aug 10 15:02:57 2021
@author: Andres Sandino
e-mail: asandino@unal.edu.co

"""

import tensorflow as tf
#import tensorflow.keras as keras

import os
from os import listdir
import math
import numpy as np

import pandas as pd

from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.metrics import FalseNegatives

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential,datasets, layers, models
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,LearningRateScheduler
from tensorflow.keras.optimizers import Adam

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#%%
#
#./data/datainfo.csv


dataframe = pd.read_csv("/home/usuario/Documentos/GitHub/DL_Project/ChexNet/data/datainfo.csv") 

dataframe_train = pd.read_csv("/home/usuario/Descargas/chexnet-master/experiments/DenseNet121/DenseNet121/train.csv")
dataframe_valid = pd.read_csv("/home/usuario/Descargas/chexnet-master/experiments/DenseNet121/DenseNet121/validation.csv")
dataframe_test = pd.read_csv("/home/usuario/Descargas/chexnet-master/experiments/DenseNet121/DenseNet121/test.csv")

#traindata = dataframe.iloc[0:1000][:]
#validdata = dataframe.iloc[10000:12000][:]
#testdata = dataframe.iloc[100000:101000][:]

# testindex = dataframetest.iloc[:,0]
# testdata2 = dataframetest.iloc[:,3:]

traindata=pd.concat([dataframe_train.iloc[:,0],dataframe_train.iloc[:,3:]],axis=1)
validdata=pd.concat([dataframe_valid.iloc[:,0],dataframe_valid.iloc[:,3:]],axis=1)
testdata=pd.concat([dataframe_test.iloc[:,0],dataframe_test.iloc[:,3:]],axis=1)

#partir en train-val
#%%
def transformar():
    def transform_batch_images(batch_x):
        batch_x = np.asarray(batch_x/255)
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        batch_x = (batch_x - imagenet_mean) / imagenet_std
        return batch_x
    return transform_batch_images

#%%
#classes =  dataframe.columns[1:].values.tolist()

thoraxlabels = ["Atelectasis","Cardiomegaly",
                "Effusion","Infiltration",
                "Mass","Nodule","Pneumonia",
                "Pneumothorax","Consolidation",
                "Edema","Emphysema","Fibrosis",
                "Pleural_Thickening","Hernia"]

classes = ['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14']

batch_size = 32
color_mode = 'rgb'  # "grayscale", "rgb", "rgba"
img_directory_path = "/home/usuario/Descargas/images/"

target_size = (224, 224)

#train_datagen = ImageDataGenerator(rescale=1./255,validation_split=validation_split)
#valid_datagen = ImageDataGenerator(rescale=1./255,validation_split=validation_split)
train_datagen = ImageDataGenerator(preprocessing_function=transformar(),
                                   horizontal_flip=True
                                   )
                                    #featurewise_std_normalization=True,
                                    #horizontal_flip=True)
valid_datagen = ImageDataGenerator(preprocessing_function=transformar(),
                                   
    
                                    )
                                   #featurewise_center=True, 
                                   #featurewise_std_normalization=True
                                    


test_datagen = ImageDataGenerator(preprocessing_function=transformar())
                                  #featurewise_center=True, 
                                  #  featurewise_std_normalization=True,
                                  #  horizontal_flip=True)

# test_datagen = ImageDataGenerator(
#                                   featurewise_center=True, 
#                                   featurewise_std_normalization=True,
                                  
#                                   )

#train_datagen = ImageDataGenerator(rescale=1./255,horizontal_flip=True)
#valid_datagen = ImageDataGenerator(rescale=1./255)



train_set = train_datagen.flow_from_dataframe(
    traindata,
    directory=img_directory_path,
    x_col="Image Index",
    y_col=thoraxlabels,
    class_mode="raw",
    target_size=target_size,
    color_mode=color_mode,
    batch_size=batch_size,
    #subset='training',
    shuffle='False',
    seed=1,
    validate_filenames=True,
    )


valid_set = valid_datagen.flow_from_dataframe(
    validdata,
    directory=img_directory_path,
    x_col="Image Index",
    y_col=thoraxlabels,
    class_mode="raw",
    target_size=target_size,
    color_mode=color_mode,
    batch_size=batch_size,
    #subset='validation',
    shuffle='False',
    seed=1,
    validate_filenames=True,
    )


test_set = test_datagen.flow_from_dataframe(
    testdata,
    directory=img_directory_path,
    x_col="Image Index",
    y_col=thoraxlabels,
    class_mode="raw",
    target_size=target_size,
    color_mode=color_mode,
    batch_size=batch_size,
    #subset='training',
    shuffle='False',
    seed=1,
    validate_filenames=True,
    )


train_steps=np.int16(train_set.samples//train_set.batch_size)
valid_steps=np.int16(valid_set.samples//test_set.batch_size)
test_steps=np.int16(test_set.samples//test_set.batch_size)

#%%

# def createmodel():

#     model = Sequential()
    
#     input_shape=(224,224,3)
    
#     model.add(tf.keras.applications.DenseNet121(include_top=True,
#                                               weights='/home/usuario/Descargas/weight.h5',
#                                               #weights='imagenet',
#                                               #weights=None,
#                                               input_tensor=None,
#                                               input_shape=(224, 224, 3),
#                                               pooling='avg',
#                                               classes=14,))
    
#     #model.add(Dense(14,activation='sigmoid'))
#     model.summary()
#     return model

# model = createmodel()

#%%

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input


input_shape = (224, 224, 3)

img_input = Input(shape=input_shape)

mdl = tf.keras.applications.DenseNet121(include_top=False,
                                              weights=None,
                                              #weights='imagenet',
                                              #weights=None,
                                              input_tensor=img_input,
                                              input_shape=(224,224,3),
                                              pooling='avg',
                                              classes=14,)


x = mdl.output
        # Last output layer is a dense connected layer with sigmoid activation according to the CheXNet paper
#predictions = Dense(len(class_names), activation="sigmoid", name="predictions")(x)
predictions = Dense(14, activation="sigmoid", name="predictions")(x)

model = Model(inputs=img_input, outputs=predictions)
# #weights_path='/home/usuario/Descargas/weight_func.h5'
weights_path='/home/usuario/Descargas/chexnet-master/experiments/DenseNet121/weight.h5'
#weights_path="/home/usuario/Descargas/modelo.h5"
model.load_weights(weights_path)


#%%
#%%# from tensorflow.keras import Model
# mdl = model.layers[0]
# numlayers = len(mdl.layers)

# SplitModel = Model(inputs=mdl.input,outputs=mdl.layers[numlayers-2].output)
# input_img = Input(shape=(224,224,3))
# SplitModelOut = SplitModel(input_img)

# output = Dense(14,activation='sigmoid')(SplitModelOut)

# model2 = Model(input_img,output)

# model2.layers[1].trainable=False

# model2.summary()


#%%

#model.load_weights('/home/usuario/Descargas/mymodel.h5')


#%%

def step_decay(epoch):
	initial_lrate = 1e-5
	drop = 0.1
	epochs_drop = 20
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

#%%


from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import utility


patience_reduce_lr=1
min_learning_rate=1e-8
RLRoP=ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience_reduce_lr,
                            verbose=1, mode="min", min_lr=min_learning_rate)


#%%

optimizer = Adam(1e-4) # 1e-5

BinaryCrossEnt=tf.keras.metrics.BinaryCrossentropy()


# tf.keras.metrics.(
#     name="binary_accuracy", dtype=None, threshold=0.5
# )

#nn=BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5)
#mm=FalseNegatives(name="fn_accuracy",thresholds=0.5)

output_directory="/home/usuario/Documentos/GitHub/DL_Project/ChexNet"
class_names=thoraxlabels
train_counts, train_pos_counts = utility.get_sample_counts(output_directory, "train", class_names)

model.compile(optimizer=optimizer, 
              loss = 'binary_crossentropy', 
              metrics=[])
#BinaryCrossEnt,

checkpoint_path = '/home/usuario/Descargas/modelo.h5'
lr = LearningRateScheduler(step_decay)
es = EarlyStopping(patience=20,mode='min', verbose=1)
mc = ModelCheckpoint(checkpoint_path, 
                     monitor='val_loss', 
                     verbose=2 , 
                     save_best_only=True, 
                     mode='min')

positive_weights_multiply=1

class_weights = utility.get_class_weights(
            train_counts,
            train_pos_counts,
            multiply=positive_weights_multiply,
        )

#%%

history = model.fit(train_set,steps_per_epoch=train_steps, validation_data=valid_set,
                                   validation_steps=valid_steps,
                                   epochs=100,verbose=1,callbacks=[mc,RLRoP])
#callbacks=[mc,lr]

#%%
#model.save_weights('/home/usuario/Descargas/mymodel.h5')
#model.save_weights('mymodel13082021.h5')


#%%


#%%

kk=[]

import matplotlib.pyplot as plt

for i in range(28,29):
    caso=i
    batch=0
    
    imgbatch = test_set[batch][0]
    
    truelabelbatch=test_set[batch][1]
    truelabel = truelabelbatch[caso]
    kk.append(truelabel)
    
    im = imgbatch[caso,:,:,:]

    # imagenet_mean = np.array([0.485,0.456,0.406])
    # imagenet_std = np.array([0.229,0.224,0.225])
    
    # im2 = (im-imagenet_mean)/imagenet_std

    #im = im2
    plt.show()
    plt.imshow(im[:,:,0],cmap='gray')
    
    im3=np.expand_dims(im,axis=0)
    
    prediction = model.predict(im3)
    prediction2 = np.squeeze(prediction,axis=0)
    #pre = np.round(prediction>0.5)
    
    
    print("case: "+np.str(i))
    print("true")
    print(truelabel)
    print("pred")
    print(np.round(prediction2))



x=[i for i in range(1,15)]
y=prediction2
# y=[0.95,
# 0.95,
# 0.89,
# 0.8,
# 0.74,
# 0.65,
# 0.59,
# 0.51,
# 0.5,
# 0.48]

plt.show()
plt.bar(x, height= y)
plt.ylim(0, 0.9)
xlocs, xlabs = plt.xticks()
xlocs=[i+1 for i in range(0,14)]
xlabs=thoraxlabels
#plt.xlabel('Max Sigma')

plt.ylabel('Prediction')
plt.xticks(xlocs, xlabs,rotation='vertical')

plt.show()


    
    
#%%

thoraxlabels = ["Atelectasis","Cardiomegaly",
                "Effusion","Infiltration",
                "Mass","Nodule","Pneumonia",
                "Pneumothorax","Consolidation",
                "Edema","Emphysema","Fibrosis",
                "Pleural_Thickening","Hernia"]



#%%

kk=[]
for i in range(test_steps+1):
    zz=test_set[i][1]
    for j in range(np.shape(zz)[0]):
        kk.append(zz[j])



truelabel=np.vstack(kk)



predictions=model.predict(test_set,test_steps+1,verbose=1)



#%%


prediction_mat=predictions


#%%



from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score, f1_score

truelabel_mat=truelabel

truevector=truelabel_mat[:,1]
predictvector=prediction_mat[:,1]

auc = roc_auc_score(truevector,predictvector)

fpr,tpr,thresholds = roc_curve(truevector,predictvector)

# plt.plot(fpr,tpr)



# tn, fp, fn, tp=confusion_matrix(truevector, np.round(predictvector>0.5)).ravel()


#%%
auc_list=[]
for i in range(14):
    auc = roc_auc_score(truelabel_mat[:,i],prediction_mat[:,i])
    fpr,tpr,thresholds = roc_curve(truelabel_mat[:,i],prediction_mat[:,i])

    plt.plot(fpr,tpr,label=i)
    auc_list.append('%.2f' % auc)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    print(auc)
    
plt.legend(auc_list)

    
#%%

from tensorflow.keras import backend as kb


def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer


img_transformed = im
class_weights = model.layers[-1].get_weights()[0]
final_conv_layer = get_output_layer(model, "bn")
get_output = kb.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])

#%%
[conv_outputs, predictions] = get_output(im3)
conv_outputs = conv_outputs[0, :, :, :]
#np.set_printoptions(threshold=np.nan)

#%%
import cv2 as cv2

index=2
cam = np.zeros(dtype=np.float32, shape=(conv_outputs.shape[:2]))
print(class_weights[index])
for i, w in enumerate(class_weights[index]):
    print(i, w)
    cam += w * conv_outputs[:, :, i]
# print(f"predictions: {predictions}")
#%%
cam /= np.max(cam)
#cam = cv2.resize(cam, img_ori.shape[:2])
cam = cv2.resize(cam, im.shape[:2])

heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap[np.where(cam < 0.2)] = 0
img = heatmap * 0.5 + im

# ratio = 1
# x1 = int(df_g["x"] * ratio)
# y1 = int(df_g["y"] * ratio)
# x2 = int((df_g["x"] + df_g["w"]) * ratio)
# y2 = int((df_g["y"] + df_g["h"]) * ratio)
# cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
# cv2.putText(img, text=label, org=(5, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#             # fontScale=0.8, color=(0, 0, 255), thickness=1)
output_path="/home/usuario/Descargas/asg.png"
cv2.imwrite(output_path, img)




