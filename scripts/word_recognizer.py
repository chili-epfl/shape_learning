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

from scipy import interpolate, signal

import argparse
parser = argparse.ArgumentParser(description='Learn a collection of letters in parallel')
parser.add_argument('word', action="store",
                help='The word to be learnt')

from pylab import matshow, show


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

    settings = (shapeType, init_datasetFile, [update_datasetFile,demo_datasetFile], datasetParam)

    return settings



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


numPoints_shapeModeler = 70

shapesLearnt = []
wordsLearnt = []
shapeLearners = []
currentWord = []
settings_shapeLearners = []
userInputCaptures = []

nb_param = 10

spaces = {}

abc = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
for letter in abc:
    (shapeType, init_datasetFile, update_datasetFiles, datasetParam) = generateSettings(letter)
    spaces[letter] = ShapeModeler(init_filename=init_datasetFile,
                                update_filenames=update_datasetFiles,
                                param_filename=datasetParam,
                                num_principle_components=nb_param)


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
            Color(1,1,1)
            d = 30.
            touch.ud['line'] = Line(points=(touch.x, touch.y))

    def on_touch_move(self, touch):
        global userShape
        touch.ud['line'].points += [touch.x, touch.y]
        userShape += [touch.x, -touch.y]

    def on_touch_up(self, touch):
        global userShape

        word = []
        total_score =0
        rest = userShape

        scan = downsampleShape(userShape, numPoints_shapeModeler,xyxyFormat=True)
        shapeCentre = ShapeModeler.getShapeCentre(scan)
        scan = np.reshape(scan, (-1, 1)); #explicitly make it 2D array with only one column
        scan = ShapeModeler.normaliseShapeHeight(np.array(scan))

        scores = np.array(separator(scan))

        scores = scores/np.max(scores)
        scores = scores*np.abs(scores)
        scores[scores<0.1]=0

        indices = np.array(range(len(scores)))
        sep = indices[scores>0]
        #print zip(sep[:-1],sep[1:])
        
        for i in sep:
            x = scan[i]
            if len(scan[scan[i:numPoints_shapeModeler]<x])>0:
                scores[i]=0
            if len(scan[scan[:i]>x])>0:
                scores[i]=0
        
        u = True
        while u:
            u = False
            sep = indices[scores>0]
            for i,j in zip(sep[:-1],sep[1:]):
                if j-i<5:
                    u = True
                    if scores[i]>scores[j]:
                        scores[j] = 0
                    else:
                        scores[i] = 0

        plt.clf()
        ShapeModeler.showShape_score(scan,scores)

        word = []

        sep = indices[scores>0]
        for i,j in zip(sep[:-1],sep[1:]):

            shape = scan[i:j+1].T.tolist()[0] + scan[numPoints_shapeModeler+i:numPoints_shapeModeler+j+1].T.tolist()[0]
            shape = downsampleShape(shape, numPoints_shapeModeler)
            shape = np.reshape(shape,(-1,1))
            shape = ShapeModeler.normaliseShapeHeight(np.array(shape))

            best_letter = '?'
            errors = {}
            values = []
            for letter in abc:
                space = spaces[letter]
                error = space.getMinDist(shape)
                errors[letter] = error
                values.append(error)

            best_letter = min(errors, key = errors.get)
            word.append(best_letter)


        print 'found ' + str(word) 
        

        print '######################'
        print ''
        userShape = []
        self.canvas.remove(touch.ud['line'])

def separator(word):

    X = word[0:numPoints_shapeModeler]
    Y = word[numPoints_shapeModeler:]

    if len(X)>3:
        scores = np.zeros(len(X))
        for i in range(len(X) - 3):

            x1 = X[i]
            y1 = Y[i]
            x2 = X[i+1]
            y2 = Y[i+1]
            x3 = X[i+2]
            y3 = Y[i+2]

            a = (x2-x1)/(np.abs(y2-y1)+0.1)
            b = (x3-x2)/(np.abs(y3-y2)+0.1)
            c = (y3-y2)-(y2-y1)

            if c<=0:
                c = 1
            else:
                c = -1

            if y1>=y2 and y2<=y3:

                scores[i]+=a
                scores[i+1]+=(a+b)*c
                scores[i+2]+=b

            if y1<=y2 and y1<=y3:

                scores[i]+=(a+b)
                scores[i+1]+=a*c
                scores[i+2]+=b

            if y3<=y1 and y3<=y2:

                scores[i]+=a
                scores[i+1]+=b*c
                scores[i+2]+=(a+b)

        return scores
    else:
        return 0




class UserInputCapture(App):

    def build(self):
        self.painter = MyPaintWidget()
        return self.painter

    def on_start(self):
        with self.painter.canvas:
            print(self.painter.width)
            Color(1, 1, 0)
            d = 30.


if __name__ == "__main__":

    plt.ion()

    try:
        UserInputCapture().run()

    except KeyboardInterrupt:
            # ShapeModeler.save()
            logger.info("Bye bye")
