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

numfile = 1
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

input_shape = (224, 224, 3)

img_input = Input(shape=input_shape)

mdl = tf.keras.applications.DenseNet121(include_top=False,
                                              weights=None,
                                              #weights='imagenet',
                                              #weights=None,
                                              input_tensor=img_input,
                                              input_shape=(224,224,3),
                                              pooling='avg',
                                              classes=15,)

x = mdl.output
        # Last output layer is a dense connected layer with sigmoid activation according to the CheXNet paper
#predictions = Dense(len(class_names), activation="sigmoid", name="predictions")(x)
predictions = Dense(15, activation="sigmoid", name="predictions")(x)

model2 = Model(inputs=img_input, outputs=predictions)

#%%

weights_path='C:/Users/Andres/Desktop/weights_model_15ft.h5'
model2.load_weights(weights_path)

#%%

model2.save('C:/Users/Andres/Desktop/ChexNetModel3.h5')

# #%%# from tensorflow.keras import Model
# #mdl = model.layers[0]
# mdl = model
# numlayers = len(mdl.layers)

# SplitModel = Model(inputs=mdl.input,outputs=mdl.layers[numlayers-2].output)
# input_img = Input(shape=(224,224,3))
# SplitModelOut = SplitModel(input_img)

# output = Dense(15,activation='sigmoid')(SplitModelOut)

# model2 = Model(input_img,output)

# weights_path='C:/Users/Andres/Desktop/modelo_15Ft.h5'
# model2.load_weights(weights_path)

# #%%

# layers = [l for l in model.layers]

# #%%
# layer_id=248
# x = layers[0].output

# #%%
# for i in range(1, len(layers)):
#     if i == layer_id:
#         x = new_layer(x)
#     else:
#         print(i)
#         x = layers[i](x)


# #%%
# model3 = tf.keras.Sequential()

# #%%

# for i in range(1,10):
#     model3.add(layers[i])

# #%%

# def replace_intermediate_layer_in_keras(model, layer_id, new_layer):
#     from tensorflow.keras.models import Model

#     layers = [l for l in model.layers]

#     x = layers[0].output
#     for i in range(1, len(layers)):
#         if i == layer_id:
#             x = new_layer(x)
#         else:
#             x = layers[i](x)

#     new_model = Model(input=layers[0].input, output=x)
#     return new_model

# def insert_intermediate_layer_in_keras(model, layer_id, new_layer):
#     from keras.models import Model

#     layers = [l for l in model.layers]

#     x = layers[0].output
#     for i in range(1, len(layers)):
#         if i == layer_id:
#             x = new_layer(x)
#         x = layers[i](x)

#     new_model = Model(input=layers[0].input, output=x)
#     return new_model

# #%%

# model4 = replace_intermediate_layer_in_keras(mdl, 248, Dense(15,activation='sigmoid'))


# #%%

# model3 = tf.keras.Sequential()

# #%%

# for layer in mdl.layers[:-1]: # go through until last layer
#     model.add(layer)



#%%

# model2.save('C:/Users/Andres/Desktop/ChexNetModel2.h5')

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
img_out = heatmap * 0.15+ im

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

# thoraxlabels = ["Atelectasis","Cardiomegaly",
#                 "Effusion","Infiltration",
#                 "Mass","Nodule","Pneumonia",
#                 "Pneumothorax","Consolidation",
#                 "Edema","Emphysema","Fibrosis",
#                 "Pleural_Thickening","Hernia"]

# df=pd.DataFrame(columns=['Labels','Predictions'])
# df['Labels'] = thoraxlabels
# df['Predictions'] = prediction
# df.sort_values(by=['Predictions'],ascending=False)

# df2 = df.iloc[:3]
# df3=df2.sort_values(by=['Predictions'],ascending=False)

# label1 = df3.iloc[0][0]
# pred1 = str(round(df3.iloc[0][1]*100,1))+" %"

# label2 = df3.iloc[1][0]
# pred2 = str(round(df3.iloc[1][1]*100,1))+" %"

# label3 = df3.iloc[2][0]
# pred3 = str(round(df3.iloc[2][1]*100,1))+" %"

#%%

thoraxlabels = ["Atelectasis",
                "Cardiomegaly",
                "Effusion",
                "Infiltration",
                "Mass",
                "Nodule",
                "Pneumonia",
                "Pneumothorax",
                "Consolidation",
                "Edema",
                "Emphysema",
                "Fibrosis",
                "Pleural_Thickening",
                "Hernia"]

ThoraxDataFrame=pd.DataFrame(columns=['Labels','Predictions'])
ThoraxDataFrame['Labels'] = thoraxlabels
ThoraxDataFrame['Predictions'] = prediction
ThoraxDataFrame.sort_values(by=['Predictions'],ascending=False)

ThoraxDataFrameSubset = ThoraxDataFrame.iloc[:3]
Sort_ThoraxDataFrame = ThoraxDataFrameSubset.sort_values(by=['Predictions'],ascending=False)

LabelList=[]
PredictionList=[]


for i in range(3):

    Label = Sort_ThoraxDataFrame.iloc[i][0]
    LabelList.append(Label)
    Prediction = str(round(Sort_ThoraxDataFrame.iloc[i][1]*100,1))+" %"
    PredictionList.append(Prediction)
        
PredListOutput = [LabelList[0],PredictionList[0],
                  LabelList[1],PredictionList[1],
                  LabelList[2],PredictionList[2]
                  ]



#%%

patient_name = "Prueba 1"
ID = '102234'
genre = 'F'
date = '02/02/02'
study_name = 'CHEST CT'
study_date = '01/01/01'

report='This report was automaticly generated by theStella services. At least one patology pattern was indentified in this study. The heatmap overlead on the image represeted the area with the AI considered to do the automatic evaluation.'
report = report +report
region = 'US'





#%%


from fpdf import FPDF
border=0
# 1. Set up the PDF doc basics
pdf = FPDF('P', 'cm', 'Letter')
#pdf.open(,'')
#pdf.set_margins(0.5,1.5,2.5)
pdf.add_page()
pdf.set_left_margin(1.5)
#pdf.image('C:/Users/Andres/Desktop/latam.jpeg',x=0,y=0,w=21.59,h=27.94)
pdf.ln(2)

pdf.set_font('Arial', 'BU',28)
pdf.set_text_color(32, 32,91)
pdf.set_fill_color(173, 197, 231)

pdf.cell(18, 1.5, 'Chest X-RAY Report', 0, 0,'C', fill=1)
pdf.set_font('Arial', '', 12)

pdf.ln(2)
pdf.cell(18, 0.5,'Patient information',align='C',border=0)

pdf.ln(0.5)
pdf.line(pdf.x, pdf.y, pdf.x+18, pdf.y)

pdf.ln(0.7)

pdf.cell(5, 0.5,'ID',align='C',border=border)

x_position = pdf.x
pdf.x = x_position
pdf.cell(5, 0.5,'Name',align='C',border=border)

x_position = pdf.x
pdf.x = x_position
pdf.cell(4, 0.5,'Birth date',align='C',border=border)

x_position = pdf.x
pdf.x = x_position
pdf.cell(4, 0.5,'Genre',align='C',border=border)

pdf.set_text_color(0,0,0)
pdf.ln(1)
pdf.cell(5, 0.5,ID,align='C',border=border)

x_position = pdf.x
pdf.x = x_position
pdf.cell(5, 0.5,patient_name,align='C',border=border)

x_position = pdf.x
pdf.x = x_position
pdf.cell(4, 0.5,date,align='C',border=border)

x_position = pdf.x
pdf.x = x_position
pdf.cell(4, 0.5,genre,align='C',border=border)

pdf.set_text_color(32, 32,91)
pdf.ln(1)
pdf.cell(18, 0.5,'Study Information',align='C',border=border)
pdf.ln(0.5)
pdf.line(pdf.x, pdf.y, pdf.x+18, pdf.y)

#pdf.ln(0.5)
pdf.cell(18, 0.5,'',align='C',border=border)
pdf.ln(0.5)

pdf.cell(5, 0.5,'Name',align='C',border=border)

x_position = pdf.x
pdf.x = x_position
pdf.cell(5, 0.5,'Date',align='C',border=border)

pdf.set_text_color(0, 0,0)
pdf.ln(1)
pdf.cell(5, 0.5,study_name,align='C',border=border)

x_position = pdf.x
pdf.x = x_position
pdf.cell(5, 0.5,study_date,align='C',border=border)


pdf.ln(1)
pdf.set_text_color(32, 32,91)
pdf.set_font('Arial', 'B', 14)
pdf.cell(10, 1,'Evaluation description',align='L',border=border)
pdf.set_text_color(0,0,0)

pdf.ln(1)
pdf.set_font('Arial', '', 10)
pdf.multi_cell(10,0.7,report,align='L',border=border)

# y_position = 10
# #pdf.y = y_position
# x_position = 15
#pdf.y = y_position

pdf.image('C:/Users/Andres/Desktop/asg.png',x=12,y=11,w=7.5,h=7.5)

y_position = 19
pdf.y = y_position

pdf.set_text_color(32, 32,91)
pdf.set_font('Arial', 'B', 14)
pdf.cell(10, 0.5,'Computer Assisted Diagnosis (CAD)',align='L',border=border)

pdf.ln(1)

pdf.set_text_color(0,0,0) #DarkBlue
pdf.set_font('Arial', '', 10)
pdf.multi_cell(10,0.7, txt=label1 + ": "+ pred1 ,align='L',border=border)
pdf.multi_cell(10,0.7, txt=label2 + ": "+ pred2 ,align='L',border=border)
pdf.multi_cell(10,0.7, txt=label3 + ": "+ pred3 ,align='L',border=border)



#pdf.image('C:/Users/Andres/Desktop/asg.png',x=1,y=1,w=10,h=10)

pdf.output('C:/Users/Andres/Desktop/fpdf_pdf_report.pdf', 'F')


from PyPDF2 import PdfFileWriter, PdfFileReader, PdfFileMerger
#from pdfrw import PageMerge, PdfReader, PdfWriter

pdf1Reader = PdfFileReader('C:/Users/Andres/Desktop/fpdf_pdf_report.pdf','rb')
pdf2Reader = PdfFileReader('C:/Users/Andres/Desktop/IMEXHSLATAM.pdf','rb')

aa=pdf1Reader.getPage(0)
bb=pdf2Reader.getPage(0)

#pdfWriter = PdfFileWriter()
pdfWriter = PdfFileMerger()
aa.mergePage(bb)
output_pdf = PdfFileWriter()
output_pdf.addPage(aa)



pdfOutputFile = open('C:/Users/Andres/Desktop/MergedFiles.pdf', 'wb')
output_pdf.write(pdfOutputFile)

pdfOutputFile.close()


"""
#%%
pdf.add_page()
#pdf.image('C:/Users/Andres/Desktop/asg.png',x=1,y=1,w=200,h=60)
pdf.image('C:/Users/Andres/Desktop/image1.png',x=0,y=0,w=200,h=10)

pdf.image('C:/Users/Andres/Desktop/imexhs/logoimex.png',x=150,y=15,w=40,h=10)
pdf.set_font('Arial', 'B', 18)
## Title
pdf.ln(15)
pdf.cell(40)

pdf.set_text_color(0,0,139) #DarkBlue


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

"""