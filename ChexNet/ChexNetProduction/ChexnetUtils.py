import pydicom as dicom
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import os
import pandas as pd


from fpdf import FPDF

from tensorflow.keras import backend as kb

from ChexnetConstantManager import ImgSize, MaxIntensityValue
from ChexnetConstantManager import OutputPath_Heatmap, HeatmapOpacity


def get_output_layer(model, layer_name):
    
    # get the symbolic outputs of each "key" layer (we gave them unique names)
    
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    
    return layer

def gradcam(model,img,img_trans2):
    
    # https://keras.io/examples/vision/grad_cam/
    
    """ Activation heatmap for an image classification model """
    
    class_weights = model.layers[-1].get_weights()[0]
    final_conv_layer = get_output_layer(model, "bn")
    get_output = kb.function([model.layers[0].input], 
                             [final_conv_layer.output, 
                              model.layers[-1].output])
    
    
    [conv_outputs, predictions] = get_output(img_trans2)
    conv_outputs = conv_outputs[0, :, :, :]
    #np.set_printoptions(threshold=np.nan)
    
    im = cv.resize(img,ImgSize)    
    
    index=np.argmax(predictions)
    cam = np.zeros(dtype=np.float32, shape=(conv_outputs.shape[:2]))
    
    for i, w in enumerate(class_weights[index]):
        cam += w * conv_outputs[:, :, i]
      
    cam /= np.max(cam)
    cam = cv.resize(cam, im.shape[:2])
    
    heatmap = cv.applyColorMap(np.uint8(255 * cam), 
                               cv.COLORMAP_JET)
    
    heatmap[np.where(cam < 0.2)] = 0
    img_out = heatmap * HeatmapOpacity + im
    output_path= OutputPath_Heatmap
    cv.imwrite(output_path, img_out)
    
    return img_out