#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval

import sys
import re
from collections import OrderedDict

from shape_learning.shape_learner_manager import ShapeLearnerManager
from shape_learning.shape_learner import SettingsStruct
from shape_learning.shape_modeler import ShapeModeler #for normaliseShapeHeight()

import os.path




from kivy.config import Config
Config.set('kivy', 'logger_enable', 0)
Config.write()

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse, Line

from scipy import interpolate

import argparse
parser = argparse.ArgumentParser(description='Learn a collection of letters in parallel')
parser.add_argument('word', action="store",
                help='The word to be learnt')

numPoints_shapeModeler = 70


#actions = re.compile('201\d-\d\d-\d\d (?P<time>\d\d:\d\d:\d\d).*INFO - (?P<letter>\w):.*')

action = re.compile('.*INFO - (?P<letter>\w):.*(?P<type>demonstration|generated|learning).*arams: (?P<params>\[[\de., -]*\]). Path: (?P<path>\[[\de., -]*\])')


actions_letters = {}
demo_letters = {}


def generateSettings(shapeType):
    paramsToVary = [3];            #Natural number between 1 and numPrincipleComponents, representing which principle component to vary from the template
    initialBounds_stdDevMultiples = np.array([[-6, 6]]);  #Starting bounds for paramToVary, as multiples of the parameter's observed standard deviation in the dataset
    doGroupwiseComparison = True; #instead of pairwise comparison with most recent two shapes
    initialParamValue = np.NaN
    initialBounds = np.array([[np.NaN, np.NaN]])

    init_datasetFile = init_datasetDirectory + '/' + shapeType + '.dat'
    update_datasetFile = update_datasetDirectory + '/' + shapeType + '.dat'
    demo_datasetFile = demo_datasetDirectory + '/' + shapeType + '.dat'
    
    if not os.path.exists(init_datasetFile):
        raise RuntimeError("Dataset not found for shape" + shapeType)
        
    if not os.path.exists(update_datasetFile):
        try:
            with open(update_datasetFile, 'w') as f:
                pass
        except IOError:
                    raise RuntimeError("no writing permission for file"+update_datasetFile)
                    
    if not os.path.exists(demo_datasetFile):
        try:
            with open(demo_datasetFile, 'w') as f:
                pass
        except IOError:
                    raise RuntimeError("no writing permission for file"+demo_datasetFile)
        
    try:
        datasetParam = init_datasetDirectory + '/params.dat'
        with open(datasetParam, 'r') as f:
            line = f.readline()
            test = line.replace('[','').replace(']\n','')==shapeType
            while test==False:
                line = f.readline()
                if line:
                    test = line.replace('[','').replace(']\n','')==shapeType
                else:
                    break
            if test:
                s = f.readline().replace('\n','')
                initialParamValue = (float)(s)
            else:
                initialParamValue = 0.0
                print("parameters not found for shape "+ shapeType +'\n'+'Default : 0.0')
            
    except IOError:
        raise RuntimeError("no reading permission for file"+datasetParam)

    settings = SettingsStruct(shape_learning = shapeType,
                                paramsToVary = paramsToVary, 
                                doGroupwiseComparison = True,
                                initDatasetFile = init_datasetFile, 
                                updateDatasetFiles = [update_datasetFile,demo_datasetFile],
                                initialBounds = initialBounds, 
                                initialBounds_stdDevMultiples = initialBounds_stdDevMultiples,
                                initialParamValue = initialParamValue, 
                                minParamDiff = 0.4)
    return settings



def showShape(shape ):
    plt.figure(1)
    plt.clf()
    ShapeModeler.normaliseAndShowShape(shape)#.path)



if __name__ == "__main__":

    with open(sys.argv[1], 'r') as log:

        for line in log.readlines():

            found = action.search(line)
            if found:
                letter = found.group('letter')
                type = found.group('type')
                params = literal_eval(found.group('params'))
                path = literal_eval(found.group('path'))
                actions_letters.setdefault(letter, []).append((type, params))
                if type=='demonstration':
                    demo_letters.setdefault(letter,[]).append(path)

    for bad_letter, value  in demo_letters.items():

        for path in value:

            plt.ion()

            userShape = path

            userShape = np.reshape(userShape, (-1, 1));
            showShape(userShape)

            letter = raw_input('letter ? ')



            import inspect
            fileName = inspect.getsourcefile(ShapeModeler)
            installDirectory = fileName.split('/lib')[0]
            #datasetDirectory = installDirectory + '/share/shape_learning/letter_model_datasets/uji_pen_chars2'
            init_datasetDirectory = installDirectory + '/share/shape_learning/letter_model_datasets/alexis_set_for_children'
            update_datasetDirectory = installDirectory + '/share/shape_learning/letter_model_datasets/alexis_set_for_children'
            demo_datasetDirectory = installDirectory + '/share/shape_learning/letter_model_datasets/diego_set'

            if not os.path.exists(init_datasetDirectory):
                raise RuntimeError("initial dataset directory not found !")
            if not os.path.exists(update_datasetDirectory):
                os.makedir(update_datasetDirectory)

            wordManager = ShapeLearnerManager(generateSettings)
            wordSeenBefore = wordManager.newCollection(letter)

            shape = wordManager.startNextShapeLearner()


            # learning part :
            print('Received demo for letter ' + letter)

            #userShape = path
            #print userShape
            #userShape = np.reshape(userShape, (-1, 1)); #explicitly make it 2D array with only one column
            shape = wordManager.respondToDemonstration(0, userShape)
            wordManager.save_all(0)







'''
for letter, value in actions_letters.items():
    print("%s: %s" % (letter, str(value)))
'''
