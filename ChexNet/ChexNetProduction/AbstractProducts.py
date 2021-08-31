import pickle
import joblib
from tensorflow.keras.models import load_model

def load_mdl_chexnet():
    path='./models/'
    mdlfilename='ChexNetModel.h5'
    mdl_chexnet=load_model(path+mdlfilename)
    return mdl_chexnet
