# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 08:44:59 2021
@author: Andres
"""

import numpy as np

"""Constants for run_preprocessing method"""

ImgSize = (224,224)
MaxIntensityValue = 255
ImagenetMean = np.array([0.485, 0.456, 0.406])
ImagenetStd = np.array([0.229, 0.224, 0.225])



# ImgIn = cv.resize(ImgIn, (224,224))    
# ImgIn = np.asarray(ImgIn/255)
# imagenet_mean = np.array([0.485, 0.456, 0.406])
# imagenet_std = np.array([0.229, 0.224, 0.225])

# ImgIn = (ImgIn - imagenet_mean) / imagenet_std





WinWidth=1500
WinLength=-500

imgnormsize = [512,512]
inputimgCNNscale = 4

SEsize = 5
#kernel = np.ones((5, 5), np.uint8)