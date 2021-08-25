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


#%%

imgdir = "C:/Users/Andres/Desktop/images/"

numfile = 1
listimgfile = os.listdir(imgdir)
imgfile = os.path.join(imgpath,listimgfile[numfile])

img = cv.imread(imgfile)

plt.imshow(img,cmap='gray')
plt.axis('off')
plt.title(listimgfile[numfile])

#%%







