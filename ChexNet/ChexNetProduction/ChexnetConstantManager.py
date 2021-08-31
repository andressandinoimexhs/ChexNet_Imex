# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 08:44:59 2021
@author: Andres
"""

import numpy as np

#Constants for run_preprocessing method

ImgSize = (224,224)
MaxIntensityValue = 255
ImagenetMean = np.array([0.485, 0.456, 0.406])
ImagenetStd = np.array([0.229, 0.224, 0.225])

#Constants for ChexnetUtils.py

ImgSize = (224,224)
MaxIntensityValue = 255

HeatmapOpacity = 0.15

OutputPath_Heatmap = "./misc/thoraxheatmap.png"