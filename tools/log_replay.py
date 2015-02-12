#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval

import sys
import re
from collections import OrderedDict

from shape_learning.shape_learner_manager import ShapeLearnerManager
from shape_learning.shape_learner import SettingsStruct
from shape_learning.shape_modeler import ShapeModeler

import os.path

import argparse
parser = argparse.ArgumentParser(description='Learn a collection of letters in parallel')
parser.add_argument('word', action="store",
                help='The word to be learnt')

action = re.compile('.*INFO - (?P<letter>\w):.*(?P<type>demonstration|generated|learning).*arams: (?P<params>\[[\de., -]*\]). Path: (?P<path>\[[\de., -]*\])')

demo_letters = {}

def generateSettings(shapeType):
    paramsToVary = [3];
    initialBounds_stdDevMultiples = np.array([[-6, 6]]);
    doGroupwiseComparison = True;
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
                u = f.readline().replace('\n','')
                initialParamValue = [(float)(s) for s in u.split(',')]
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
                                paramFile = datasetParam,
                                initialBounds = initialBounds, 
                                initialBounds_stdDevMultiples = initialBounds_stdDevMultiples,
                                initialParamValue = initialParamValue, 
                                minParamDiff = 0.4)
    return settings


def showShape(shape ):
    plt.figure(1)
    plt.clf()
    ShapeModeler.normaliseAndShowShape(shape)

if __name__ == "__main__":

    plt.ion()

    with open(sys.argv[1], 'r') as log:

        for line in log.readlines():

            found = action.search(line)
            if found:
                letter = found.group('letter')
                type = found.group('type')
                params = literal_eval(found.group('params'))
                path = literal_eval(found.group('path'))
                if type=='demonstration':
                    demo_letters.setdefault(letter,[]).append(path)

                    # if there is a bug inside the log file, e.g. the letter dont match with the shape
                    # then we want to show the shape and enter by hand the good letter :
                    #-------------------------------------------------------------------
                    #userShape = path
                    #userShape = np.reshape(userShape, (-1, 1))
                    #showShape(userShape)
                    #letter = raw_input('letter ? ')

    for letter, value  in demo_letters.items():

        for path in value:

            userShape = path
            userShape = np.reshape(userShape, (-1, 1))

            import inspect
            fileName = inspect.getsourcefile(ShapeModeler)
            installDirectory = fileName.split('/lib')[0]
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

            # learning :
            print('Received demo for letter ' + letter)
            shape = wordManager.respondToDemonstration(0, userShape)
            wordManager.save_all(0)
            wordManager.save_params(0)
