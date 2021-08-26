# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 11:40:37 2021

@author: Andres
"""


import tensorflow as tf
#import tensorflow.keras as keras

import os
from os import listdir
from os.path import join


import math
import numpy as np
import pandas as pd

import cv2 as cv


import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Input


#%%

imgdir = "C:/Users/Andres/Desktop/images/"

numfile = 50
listimgfile = os.listdir(imgdir)
imgfile = os.path.join(imgdir,listimgfile[numfile])

img = cv.imread(imgfile)

plt.imshow(img,cmap='gray')
plt.axis('off')
plt.title(listimgfile[numfile])

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
weights_path='C:/Users/Andres/Desktop/weight_func.h5'
#weights_path="/home/usuario/Descargas/modelo.h5"
model.load_weights(weights_path)

#%%

def imgpreprocessing(batch_x):
    
    batch_x = cv.resize(batch_x, (224,224))    
    batch_x = np.asarray(batch_x/255)
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    batch_x = (batch_x - imagenet_mean) / imagenet_std

    return batch_x


img_trans = imgpreprocessing(img)
img_trans2 = np.expand_dims(img_trans,axis=0)

prediction = model.predict(img_trans2)
prediction = np.squeeze(prediction,axis=0)


#%%
ll=[]
for i in enumerate(prediction):
    ll.append(i)



#%%


from tensorflow.keras import backend as kb


def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer

img_transformed = img_trans2
class_weights = model.layers[-1].get_weights()[0]
final_conv_layer = get_output_layer(model, "bn")
get_output = kb.function([model.layers[0].input], 
                         [final_conv_layer.output, 
                          model.layers[-1].output])


[conv_outputs, predictions] = get_output(img_trans2)
conv_outputs = conv_outputs[0, :, :, :]
#np.set_printoptions(threshold=np.nan)

im = cv.resize(img, (224,224))    

index=np.argmax(predictions)
cam = np.zeros(dtype=np.float32, shape=(conv_outputs.shape[:2]))
print(class_weights[index])
for i, w in enumerate(class_weights[index]):
    print(i, w)
    cam += w * conv_outputs[:, :, i]
# print(f"predictions: {predictions}")

cam /= np.max(cam)
#cam = cv2.resize(cam, img_ori.shape[:2])
cam = cv.resize(cam, im.shape[:2])

heatmap = cv.applyColorMap(np.uint8(255 * cam), cv.COLORMAP_JET)
heatmap[np.where(cam < 0.2)] = 0
img_out = heatmap * 0.1+ im

# ratio = 1
# x1 = int(df_g["x"] * ratio)
# y1 = int(df_g["y"] * ratio)
# x2 = int((df_g["x"] + df_g["w"]) * ratio)
# y2 = int((df_g["y"] + df_g["h"]) * ratio)
# cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
# cv2.putText(img, text=label, org=(5, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#             # fontScale=0.8, color=(0, 0, 255), thickness=1)
output_path="C:/Users/Andres/Desktop/asg.png"
cv.imwrite(output_path, img_out)


#%%

thoraxlabels = ["Atelectasis","Cardiomegaly",
                "Effusion","Infiltration",
                "Mass","Nodule","Pneumonia",
                "Pneumothorax","Consolidation",
                "Edema","Emphysema","Fibrosis",
                "Pleural_Thickening","Hernia"]

df=pd.DataFrame(columns=['Labels','Predictions'])
df['Labels'] = thoraxlabels
df['Predictions'] = prediction
df.sort_values(by=['Predictions'],ascending=False)

df2 = df.iloc[:3]
df3=df2.sort_values(by=['Predictions'],ascending=False)

label1 = df3.iloc[0][0]
pred1 = str(round(df3.iloc[0][1]*100,1))+" %"

label2 = df3.iloc[1][0]
pred2 = str(round(df3.iloc[1][1]*100,1))+" %"

label3 = df3.iloc[2][0]
pred3 = str(round(df3.iloc[2][1]*100,1))+" %"


#%%

report='This report was automaticly generated by theStella services. At least one patology pattern was indentified in this study. The heatmap overlead on the image represeted the area with the AI considered to do the automatic evaluation.'

from fpdf import FPDF

# 1. Set up the PDF doc basics
pdf = FPDF()
pdf.set_margins(25,15,25)
pdf.add_page()
#pdf.image('C:/Users/Andres/Desktop/asg.png',x=1,y=1,w=200,h=60)


pdf.image('C:/Users/Andres/Desktop/imexhs/logoimex.png',x=150,y=15,w=40,h=10)
pdf.set_font('Arial', 'B', 18)
## Title
pdf.ln(15)
pdf.cell(40)

pdf.set_text_color(0,0,139) #DarkBlue
pdf.set_font('Arial', 'BU', 18)
pdf.set_fill_color(173, 216, 230)
pdf.cell(100, 15, 'Chest X-RAY Report', 0, 0,'C', fill=1)

pdf.ln(20)
pdf.set_text_color(0,0,0) #DarkBlue
pdf.set_font('Arial', 'B', 14)
pdf.multi_cell(100,10, txt='Patient information' ,align='L',border=0)
pdf.set_font('Arial', 'B', 12)
pdf.multi_cell(100,8, "ID:",border=0)
pdf.multi_cell(100,8, "Name:" ,align='L',border=0)
pdf.multi_cell(100,8, "Age:" ,align='L',border=0)
pdf.multi_cell(100,8, "Genre:" ,align='L',border=0)
top = pdf.y
pdf.y = top+10

pdf.line(40, top+2, 170, top+2)
#pdf.cell(20, 20, 'Report')
## Line breaks
## Image
pdf.image('C:/Users/Andres/Desktop/asg.png',x=130,y=top+10,w=60,h=60)

pdf.set_text_color(0,0,139) #DarkBlue
pdf.set_font('Arial', 'BU', 14)

pdf.multi_cell(100,10, txt='Evalution description' ,align='L',border=0)

pdf.set_text_color(0,0,0) #Black
pdf.set_font('Arial', 'B', 12)
pdf.multi_cell(100,50, txt="Description Here",align='L',border=0)

top = pdf.y
pdf.y = top+20

pdf.line(40, top+10, 170, top+10)

pdf.set_fill_color(173, 216, 230)
pdf.set_font('Arial', 'BU', 14)
pdf.set_text_color(0,0,139) #DarkBlue
pdf.cell(100,10, txt='Computer Assisted Diagnosis (CAD)' ,align='L',border=0)

pdf.ln(12)

pdf.set_text_color(0,0,0) #DarkBlue
pdf.set_font('Arial', 'B', 12)
pdf.multi_cell(100,6, txt=label1 + ": "+ pred1 ,align='L',border=0)
pdf.multi_cell(100,6, txt=label2 + ": "+ pred2 ,align='L',border=0)
pdf.multi_cell(100,6, txt=label3 + ": "+ pred3 ,align='L',border=0)
pdf.ln(20)


pdf.footer()

# top = pdf.y

# # Calculate x position of next cell
# offset = pdf.x + 60

# pdf.multi_cell(40,40,'Hello World!,how are you today',1,0)

# # Reset y coordinate
# pdf.y = top

# # Move to computed offset
# pdf.x = offset 

# pdf.multi_cell(40,40,'This cell needs to beside the other',1,0)
# pdf.image('C:/Users/Andres/Desktop/asg.png')

# pdf.ln(7)
# pdf.cell(100, 10, label1 + ": "+ pred1)
# pdf.ln(7)
# pdf.cell(100, 10, label2 + ": "+ pred2)
# pdf.ln(7)
# pdf.cell(100, 10, label3 + ": "+ pred3)



# ## Line breaks
# pdf.ln(20)

#output_df_to_pdf(pdf, sp500_history_summary_pdf)
# 3. Output the PDF file
pdf.output('C:/Users/Andres/Desktop/fpdf_pdf_report.pdf', 'F')
