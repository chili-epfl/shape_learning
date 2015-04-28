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


multi_stroke_letters = ['i','j','t']

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

def uniformize_xxyy(shape):

    if len(shape)>2:
        # comput length of the shape:
        shape_length = 0
        numPointsInShape = len(shape)/2
        last_x = shape[0]
        last_y = shape[numPointsInShape]
        scale = [0]
        for i in range(numPointsInShape-1):
            x = shape[i + 1]
            y = shape[i + numPointsInShape + 1]
            last_x = shape[i]
            last_y = shape[i + numPointsInShape]
            shape_length += np.sqrt((x-last_x)**2 + (y-last_y)**2)
            #last_x = x
            #last_y = y
            scale.append(shape_length)

        # find new points:
        new_shape_x = []
        new_shape_y = []
        step = shape_length/float(numPointsInShape)
        biggest_smoller_point = 0
        new_shape_x.append(shape[0])
        new_shape_y.append(shape[numPointsInShape])
        for i in 1+np.array(range(numPointsInShape)):
            while i*step > scale[biggest_smoller_point]:
                biggest_smoller_point += 1
            biggest_smoller_point -= 1
            x0 = shape[biggest_smoller_point]
            y0 = shape[biggest_smoller_point+numPointsInShape]
            x1 = shape[biggest_smoller_point+1]
            y1 = shape[biggest_smoller_point+1+numPointsInShape]
            diff = float(i*step-scale[biggest_smoller_point])
            dist = float(scale[biggest_smoller_point+1]-scale[biggest_smoller_point])
            new_x = x0 + diff*(x1-x0)/dist
            new_y = y0 + diff*(y1-y0)/dist
            new_shape_x.append(new_x)
            new_shape_y.append(new_y)
        new_shape_x.append(shape[numPointsInShape-1])
        new_shape_y.append(shape[-1])

        return new_shape_x + new_shape_y

    else:
        return shape



def uniformize_xyxy(shape):

    if len(shape)>2:
        # comput length of the shape:
        shape_length = 0
        numPointsInShape = len(shape)/2
        last_x = shape[0]
        last_y = shape[1]
        scale = [0]
        for i in range(numPointsInShape-1):
            x = shape[2*(i+1)]
            y = shape[2*(i+1) + 1]
            last_x = shape[2*i]
            last_y = shape[2*i + 1]
            shape_length += np.sqrt((x-last_x)**2 + (y-last_y)**2)
            #last_x = x
            #last_y = y
            scale.append(shape_length)

        # find new points:
        new_shape = []
        step = shape_length/float(numPointsInShape)
        biggest_smoller_point = 0
        new_shape.append(shape[0])
        new_shape.append(shape[1])
        for i in 1+np.array(range(numPointsInShape-1)):
            while i*step > scale[biggest_smoller_point]:
                biggest_smoller_point += 1
            biggest_smoller_point -= 1

            x0 = shape[2*biggest_smoller_point]
            y0 = shape[2*biggest_smoller_point + 1]
            x1 = shape[2*(biggest_smoller_point+1)]
            y1 = shape[2*(biggest_smoller_point+1) + 1]
            diff = float(i*step - scale[biggest_smoller_point])
            dist = float(scale[biggest_smoller_point+1]-scale[biggest_smoller_point])
            new_x = x0 + diff*(x1-x0)/dist
            new_y = y0 + diff*(y1-y0)/dist
            new_shape.append(new_x)
            new_shape.append(new_y)

        new_shape.append(shape[-2])
        new_shape.append(shape[-1])

        return new_shape

    else:
        return shape




userShape = []
lastStroke = []
mainStroke = []
class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        with self.canvas:

            #self.canvas.clear()
            Color(1, 1, 0)
            d = 30.
            touch.ud['line'] = Line(points=(touch.x, touch.y))

    def on_touch_move(self, touch):
        global lastStroke
        touch.ud['line'].points += [touch.x, touch.y]
        lastStroke += [touch.x, -touch.y]

    def on_touch_up(self, touch):
        global lastStroke
        global mainStroke
        global userShape
        touch.ud['line'].points
        
        print lastStroke
        lastStroke = uniformize_xyxy(lastStroke)
        print lastStroke
        print "..........."

        if mainStroke and (len(lastStroke) > 10):

            # get the max of the usual distances between points :
            x_shape = lastStroke[0::2]
            y_shape = lastStroke[1::2]
            x_zip = np.array(zip(x_shape[1:],x_shape[:-1]))
            y_zip = np.array(zip(y_shape[1:],y_shape[:-1]))
            dists = np.diff(x_zip)**2 + np.diff(y_zip)**2
            dist_max = max(dists)

            # starting point:
            x1 = lastStroke[0]
            y1 = lastStroke[1]
            # ending point:
            xn = lastStroke[-2]
            yn = lastStroke[-1]

            # check if the starting point is close enought to the main shape :
            x_main = np.array(mainStroke[0::2])
            y_main =  np.array(mainStroke[1::2])
            if len(x_main)==len(y_main)+1:
                x_main = x_main[:-1]
            if len(y_main)==len(x_main)+1:
                y_main = y_main[:-1]

            start_dists = (x_main-x1)**2 + (y_main-y1)**2
            dist_min_to_main = min(start_dists)
            correction = dist_min_to_main <= 2*dist_max
 
            print dist_min_to_main
            print dist_max
            print "---"

            independant_stroke = dist_min_to_main > 2*dist_max

            if correction:

                corrected_stroke = []

                # find the indice where the main stroke is broken :
                break_indice = np.argmin(start_dists)

                if break_indice==0:
                    corrected_stroke = lastStroke
                    print 'bug : breakindice0'
                else:
                    shape_indice = break_indice*2
                    corrected_stroke = mainStroke[0:shape_indice] + lastStroke

                # check if the ending point is close enought to the main shape :
                #x_main = np.array(mainStroke[0::2])
                #y_main =  np.array(mainStroke[1::2])
                end_dists = (x_main-xn)**2 + (y_main-yn)**2
                dist_min_to_main = min(end_dists)
                rejoining = dist_min_to_main <= 2*dist_max

                if rejoining:

                    print 'rejoining !!'
                    # find the indice where the main stroke is rejoined :
                    rejoin_indice = np.argmin(end_dists)
                    print rejoin_indice

                    if rejoin_indice < len(end_dists)-1:
                        shape_indice = rejoin_indice*2 
                        corrected_stroke += mainStroke[shape_indice:]

                mainStroke = corrected_stroke

            if independant_stroke:

                # ignoring for the moment
                print "new independant stroke, ignored !"


            #userShape.append(downsampleShape(userStroke,numPoints_shapeModeler,xyxyFormat=True))
            lastStroke = []

        elif len(lastStroke) > 10:
            mainStroke = lastStroke
            lastStroke = []

        if touch.is_double_tap:

            #userShape.append(downsampleShape(mainStroke,numPoints_shapeModeler,xyxyFormat=True))
            #userShape = userShape[:-2]

            userShape = downsampleShape(mainStroke,numPoints_shapeModeler,xyxyFormat=True)
            userShape = np.reshape(userShape, (-1, 1))

            shapeType = wordManager.shapeAtIndexInCurrentCollection(0)

            """
            if shapeType in multi_stroke_letters:
                Here manage multi_stroke_letters

            else:
                x1 = 
            """

            print('Received demo')

            '''
            for i in range(len(userShape)):
                userShape[i] = np.reshape(userShape[i], (-1, 1)); #explicitly make it 2D array with only one column
            '''

            # !!! shape.path is differnent now !
            shape = wordManager.respondToDemonstration(0, userShape)
            wordManager.save_all(0)

            userShape = []
            mainStroke = []
            #self.canvas.remove(touch.ud['line'])
            self.canvas.clear()
            
            showShape(shape, 0)

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
                                initialBounds = initialBounds, 
                                initialBounds_stdDevMultiples = initialBounds_stdDevMultiples,
                                initialParamValue = initialParamValue, 
                                minParamDiff = 0.4)
    return settings

def showShape(shape, shapeIndex):
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
    init_datasetDirectory = installDirectory + '/share/shape_learning/letter_model_datasets/alexis_set_for_children'
    update_datasetDirectory = installDirectory + '/share/shape_learning/letter_model_datasets/alexis_set_for_children'
    demo_datasetDirectory = installDirectory + '/share/shape_learning/letter_model_datasets/diego_set'
    
    if not os.path.exists(init_datasetDirectory):
        raise RuntimeError("initial dataset directory not found !")
    if not os.path.exists(update_datasetDirectory):
        os.makedir(update_datasetDirectory)

    wordManager = ShapeLearnerManager(generateSettings)
    #print wordToLearn
    #print [wordToLearn]
    wordSeenBefore = wordManager.newCollection(wordToLearn)


    plt.ion()
    for i in range(len(wordToLearn)):
        shape = wordManager.startNextShapeLearner()
        showShape(shape, i)

    try:
        UserInputCapture().run()
        
    except KeyboardInterrupt:
            # ShapeModeler.save()
            logger.info("Bye bye")
