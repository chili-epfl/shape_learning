#!/usr/bin/env python
# coding: utf-8

'''
This script just writes the shape of the letter the user draws at the top of the database.
This top-letter will be used as the optimal reference letter. We want childs to learn this letter
by playing with the robot. 
'''

from shape_learning.shape_learner_manager import ShapeLearnerManager
from shape_learning.shape_learner import SettingsStruct
from shape_learning.shape_modeler import ShapeModeler #for normaliseShapeHeight()
import os.path
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
dataSetFile = ''

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
datasetFile = ''
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

        shapeType = letter
            
        print('Received reference for letter ' + shapeType)

        userShape = numpy.reshape(userShape, (-1, 1));
        userShape = ShapeModeler.normaliseShapeHeight(numpy.array(userShape))
        
        filename = dataSetFile
        print('saving in'+filename)
        
        # scan the dataset :
        lines = []
        try: 
            with open(filename, 'r') as f:
                lines.append(f.readline())
                nb_samples =  int(lines[0].strip())
                for i in range(nb_samples+1):
                    lines.append(f.readline())
        except IOError:
            raise RuntimeError("no reading permission for file"+filename)
            
        nb_samples+=1
        
        # past the dataset :
        try:
            with open(filename, 'w') as f:
                f.write('nb_sample:\n')
                f.write('%i\n'%nb_samples)
                f.write('nb_pts:\n')
                f.write('70\n')
                f.write('ref:\n')
                f.write(' '.join(map(str,userShape.T[0]))+'\n')
                f.write('...\n')
                for i in range(nb_samples-1):
                    f.write(lines[i+2])
        except IOError:
            raise RuntimeError("no writing permission for file"+filename)

        userShape = []
        self.canvas.remove(touch.ud['line'])

class UserInputCapture(App):

    def build(self):
        self.painter = MyPaintWidget()
        return self.painter

    def on_start(self):
        with self.painter.canvas:
            print(self.painter.width)
            Color(1, 1, 0)
            d = 30.
            
            for i in range(len(letter)-1):
                x = (self.painter.width/len(letter))*(i+1)
                Line(points=(x, 0, x, self.painter.height))


if __name__ == "__main__":
    global dataSetFile
    
    #parse arguments
    args = parser.parse_args()
    letter = args.word

    import inspect
    fileName = inspect.getsourcefile(ShapeModeler)
    installDirectory = fileName.split('/lib')[0]
    datasetDirectory = installDirectory + '/share/shape_learning/letter_model_datasets/alexis_set_for_children'
    if not os.path.exists(datasetDirectory):
        raise RuntimeError("dataset directory not found !")
        
    dataSetFile = datasetDirectory + '/' + letter + '.dat'

    plt.ion()
    UserInputCapture().run()
