#!/usr/bin/env python
# coding: utf-8

from shape_learning.shape_learner_manager import ShapeLearnerManager
from shape_learning.shape_learner import SettingsStruct
from shape_learning.shape_modeler import ShapeModeler #for normaliseShapeHeight()


import numpy
import matplotlib.pyplot as plt

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

shapesLearnt = []
wordsLearnt = []
shapeLearners = []
currentWord = []
settings_shapeLearners = []
userInputCaptures = []

def downsampleShape(shape,numDesiredPoints,xyxyFormat=False):
    numPointsInShape = len(shape)/2
    if(xyxyFormat):
        #make xyxy format
        x_shape = shape[0::2]
        y_shape = shape[1::2]
    else:
        x_shape = shape[0:numPointsInShape]
        y_shape = shape[numPointsInShape:]

    if isinstance(x_shape,numpy.ndarray): #convert arrays to lists for interp1d
        x_shape = (x_shape.T).tolist()[0]
        y_shape = (y_shape.T).tolist()[0]

    #make shape have the same number of points as the shape_modeler
    t_current = numpy.linspace(0, 1, numPointsInShape)
    t_desired = numpy.linspace(0, 1, numDesiredPoints)
    f = interpolate.interp1d(t_current, x_shape, kind='cubic')
    x_shape = f(t_desired)
    f = interpolate.interp1d(t_current, y_shape, kind='cubic')
    y_shape = f(t_desired)

    shape = []
    shape[0:numPoints_shapeModeler] = x_shape
    shape[numPoints_shapeModeler:] = y_shape

    return shape


userShape = []
class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        with self.canvas:

            #self.canvas.clear()
            Color(1, 1, 0)
            d = 30.
            touch.ud['line'] = Line(points=(touch.x, touch.y))

    def on_touch_move(self, touch):
        global userShape
        touch.ud['line'].points += [touch.x, touch.y]
        userShape += [touch.x, -touch.y]

    def on_touch_up(self, touch):
        global userShape
        touch.ud['line'].points
        userShape = downsampleShape(userShape,numPoints_shapeModeler,xyxyFormat=True)

        shapeCentre = ShapeModeler.getShapeCentre(userShape)
        
        for i in range(len(wordToLearn)):
            if(shapeCentre[0] > (self.width/len(wordToLearn))*i):
                shapeIndex_demoFor = i

        shapeType = wordManager.shapeAtIndexInCurrentCollection(shapeIndex_demoFor)
            
        print('Received demo for letter ' + shapeType)

        userShape = numpy.reshape(userShape, (-1, 1)); #explicitly make it 2D array with only one column
        userShape = ShapeModeler.normaliseShapeHeight(numpy.array(userShape))
        
        #------------This is to add the user's drawing to the data base:
        dataShape = ''
        for x in userShape:
            dataShape += ('%f'%x[0] + ' ')
        dataShape += '\n'
        
        try:
            datasetFile = datasetDirectory + '/' + shapeType + '.dat'
            
            with open(datasetFile, 'rb') as f:
                l = f.readlines()[0]
                numExemples = (int)(l)
                l = '%i\n' % (numExemples+1)
                newData = l
            with open(datasetFile, 'rb') as f:
                ind = 0
                for l in f.readlines():
                    if ind>0:
                        newData = newData + l
                    ind+=1
            with open(datasetFile, 'wb') as f:
                f.write(newData)
                f.write(dataShape)
                
        except IOError:
            raise RuntimeError("Dataset not found for shape "+ shapeType)
            
        #cd---------------------------------------------------------------

        shape = wordManager.respondToDemonstration(shapeIndex_demoFor, userShape)

        userShape = []
        self.canvas.remove(touch.ud['line'])
        showShape(shape, shapeIndex_demoFor)

class UserInputCapture(App):

    def build(self):
        self.painter = MyPaintWidget()
        return self.painter

    def on_start(self):
        with self.painter.canvas:
            print(self.painter.width)
            Color(1, 1, 0)
            d = 30.
            
            for i in range(len(wordToLearn)-1):
                x = (self.painter.width/len(wordToLearn))*(i+1)
                Line(points=(x, 0, x, self.painter.height))



###---------------------------------------------- WORD LEARNING SETTINGS

def generateSettings(shapeType):
    paramsToVary = [3];            #Natural number between 1 and numPrincipleComponents, representing which principle component to vary from the template
    initialBounds_stdDevMultiples = numpy.array([[-6, 6]]);  #Starting bounds for paramToVary, as multiples of the parameter's observed standard deviation in the dataset
    doGroupwiseComparison = True; #instead of pairwise comparison with most recent two shapes
    initialParamValue = numpy.NaN
    initialBounds = numpy.array([[numpy.NaN, numpy.NaN]])
    
    try:
        datasetFile = datasetDirectory + '/' + shapeType + '.dat'
        with open(datasetFile, 'rb') as f:
            pass
    except IOError:
        raise RuntimeError("Dataset not found for shape "+ shapeType)

    settings = SettingsStruct(shape_learning = shapeType,
            paramsToVary = paramsToVary, doGroupwiseComparison = True, 
            datasetFile = datasetFile, initialBounds = initialBounds, 
            initialBounds_stdDevMultiples = initialBounds_stdDevMultiples,
            initialParamValue = initialParamValue, minParamDiff = 0.4)
    return settings

def showShape(shape,shapeIndex ):
    plt.figure(shapeIndex+1)
    plt.clf()
    ShapeModeler.normaliseAndShowShape(shape.path)

if __name__ == "__main__":
    #parse arguments
    args = parser.parse_args()
    wordToLearn = args.word

    import inspect
    fileName = inspect.getsourcefile(ShapeModeler)
    installDirectory = fileName.split('/lib')[0]
    #datasetDirectory = installDirectory + '/share/shape_learning/letter_model_datasets/uji_pen_chars2'
    datasetDirectory = installDirectory + '/share/shape_learning/letter_model_datasets/alexis_set_for_children'


    wordManager = ShapeLearnerManager(generateSettings)
    wordSeenBefore = wordManager.newCollection(wordToLearn)


    plt.ion()
    for i in range(len(wordToLearn)):
        shape = wordManager.startNextShapeLearner()
        showShape(shape, i)

    UserInputCapture().run()
