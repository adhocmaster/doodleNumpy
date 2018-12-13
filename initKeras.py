import glob
import numpy as np
import os
import warnings
import logging, sys, math
import matplotlib.pyplot as plt
from importlib import reload
import seaborn as sns
sns.set()
from datetime import datetime
import json

from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import activations
from keras import metrics

from sklearn.model_selection import train_test_split

logging.warning( "Keras dependencies loaded" )


if sys.modules.get( 'dataUtils.dataUtils', False ) != False :
    del sys.modules['dataUtils.dataUtils'] 
import dataUtils
reload(dataUtils) 
from dataUtils import dataUtils
logging.warning( "dataUtils loaded" )

if sys.modules.get( 'DoodleModels.DoodleModels', False ) != False :
    del sys.modules['DoodleModels.DoodleModels'] 
import DoodleModels
reload(DoodleModels) 
from DoodleModels import DoodleModels
logging.warning( "DoodleModels loaded" )

if sys.modules.get( 'ResNet', False ) != False :
    del sys.modules['ResNet'] 
import ResNet
reload(ResNet) 
from ResNet import ResNet
logging.warning( "ResNet loaded" )

if sys.modules.get( 'ResNet2', False ) != False :
    del sys.modules['ResNet2'] 
import ResNet2
reload(ResNet2) 
from ResNet2 import ResnetBuilder
logging.warning( "ResnetBuilder loaded" )