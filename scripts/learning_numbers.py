
#!/usr/bin/env python
# coding: utf-8

from shape_learning.shape_learner_manager import ShapeLearnerManager
from shape_learning.shape_learner import SettingsStruct
from shape_learning.shape_modeler import ShapeModeler #for normaliseShapeHeight()

import os.path

import numpy as np
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

    if isinstance(x_shape,np.ndarray): #convert arrays to lists for interp1d
        x_shape = (x_shape.T).tolist()[0]
        y_shape = (y_shape.T).tolist()[0]

    #make shape have the same number of points as the shape_modeler
    t_current = np.linspace(0, 1, numPointsInShape)
    t_desired = np.linspace(0, 1, numDesiredPoints)
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

        userShape = np.reshape(userShape, (-1, 1)); #explicitly make it 2D array with only one column
        #shape = wordManager.respondToDemonstration(shapeIndex_demoFor, userShape)
        if shapeType in ['0','8','6']:
            shape,response = wordManager.respondToGoodDemonstration_modulo_phase(shapeIndex_demoFor, userShape)
            if response:
                print 'good shape, accepted !'
            else:
                print 'wrong shape, fefused !'
        else:
            shape,response = wordManager.respondToGoodDemonstration(shapeIndex_demoFor, userShape)
            if response==0:
                print 'wrong shape, refused !'
            if response==1:
                print 'good shape, but wrong way to draw it, refused !'
            if response==2:
                print 'good shape, accepted !'
        
        #wordManager.save_all(shapeIndex_demoFor)
        wordManager.save_robot_try(shapeIndex_demoFor)

        

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
    paramsToVary = [3];
    initialBounds_stdDevMultiples = np.array([[-6, 6]]);
    doGroupwiseComparison = True;
    initialParamValue = np.NaN
    initialBounds = np.array([[np.NaN, np.NaN]])
    init_datasetFile = init_datasetDirectory + '/' + shapeType + '.dat'
    update_datasetFile = update_datasetDirectory + '/' + shapeType + '.dat'
    demo_datasetFile = demo_datasetDirectory + '/' + shapeType + '.dat'
    robot_datasetFile = robot_datasetDirectory + '/' + shapeType + '.dat'

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

    if not os.path.exists(robot_datasetFile):
        try:
            with open(robot_datasetFile, 'w') as f:
                pass
        except IOError:
                    raise RuntimeError("no writing permission for file"+robot_datasetFile)

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
                initialParamValue = [0.0,0.0,0.0,0.0,0.0]
                print("parameters not found for shape "+ shapeType +'\n'+'Default : 0.0')

    except IOError:
        raise RuntimeError("no reading permission for file"+datasetParam)

    settings = SettingsStruct(shape_learning = shapeType,
                                paramsToVary = paramsToVary, 
                                doGroupwiseComparison = True,
                                initDatasetFile = init_datasetFile,
                                updateDatasetFiles = [update_datasetFile,demo_datasetFile],
                                paramFile = datasetParam,
                                robotDataFiles = [robot_datasetFile],
                                initialBounds = initialBounds, 
                                initialBounds_stdDevMultiples = initialBounds_stdDevMultiples,
                                initialParamValue = initialParamValue, 
                                minParamDiff = 0.4)
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
    init_datasetDirectory = installDirectory + '/share/shape_learning/letter_model_datasets/bad_letters'
    update_datasetDirectory = installDirectory + '/share/shape_learning/letter_model_datasets/bad_letters'
    demo_datasetDirectory = installDirectory + '/share/shape_learning/letter_model_datasets/diego_set'
    robot_datasetDirectory = installDirectory + '/share/shape_learning/robot_tries/with_henry'
    
    if not os.path.exists(init_datasetDirectory):
        raise RuntimeError("initial dataset directory not found !")
    if not os.path.exists(update_datasetDirectory):
        os.makedir(update_datasetDirectory)

    wordManager = ShapeLearnerManager(generateSettings)
    wordSeenBefore = wordManager.newCollection(wordToLearn)


    plt.ion()
    for i in range(len(wordToLearn)):
        shape = wordManager.startNextShapeLearner()
        showShape(shape, i)

    try:
        UserInputCapture().run()
        
    except KeyboardInterrupt:
        logger.info("Bye bye")
        # ShapeModeler.save()
