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
init_datasetDirectory = installDirectory + '/share/shape_learning/letter_model_datasets/new_dat'
update_datasetDirectory = installDirectory + '/share/shape_learning/letter_model_datasets/new_dat'
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

abc = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','r#','r@'\
        ,'v#','v@','on','om','oe','ll']
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
    f = interpolate.interp1d(t_current, x_shape)#, kind='cubic')
    x_shape = f(t_desired)
    f = interpolate.interp1d(t_current, y_shape)#, kind='cubic')
    y_shape = f(t_desired)

    shape = []
    shape[0:numDesiredPoints] = x_shape
    shape[numDesiredPoints:] = y_shape

    return shape

def xyxy_to_xxyy(shape):
    numPointsInShape = len(shape)/2

    x_shape = shape[0::2]
    y_shape = shape[1::2]

    if isinstance(x_shape,np.ndarray): #convert arrays to lists for interp1d
        x_shape = (x_shape.T).tolist()[0]
        y_shape = (y_shape.T).tolist()[0]

    t_current = np.linspace(0, 1, numPointsInShape)
    f = interpolate.interp1d(t_current, x_shape)#, kind='cubic')
    x_shape = f(t_current)
    f = interpolate.interp1d(t_current, y_shape)#, kind='cubic')
    y_shape = f(t_current)

    shape = []
    shape[0:numPointsInShape] = x_shape
    shape[numPointsInShape:] = y_shape

    return shape

userShape = []
newUserShape = []
gaps = []

letter_len = 0

words = []
x_pos = []
y_pos_min = []

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global is_right
        global gaps
        global letter_len

        with self.canvas:

            #self.canvas.clear()
            Color(1,1,1)
            d = 30.
            touch.ud['line'] = Line(points=(touch.x, touch.y))

        if len(gaps)%2==1:
            gaps += [(touch.x, -touch.y)]


    def on_touch_move(self, touch):
        global userShape
        global newUserShape

        touch.ud['line'].points += [touch.x, touch.y]
        userShape += [touch.x, -touch.y]
        newUserShape += [touch.x, -touch.y]


    def on_touch_up(self, touch):
        global gaps
        global letter_len
        global userShape
        global newUserShape
        global words

        if touch.is_double_tap:

            if len(gaps)>4:
                gaps = gaps[:-4]
            else:
                gaps = []

            if len(userShape)>4:
                letter_len, word = self.read(userShape)
                words += word
                print words
                gaps = []
                userShape = []
                newUserShape = []

            else:
                print 'not enought points to read a word !'
                gaps = []
                userShape = []
                newUserShape = []

            self.canvas.clear()
            words = []


        else:

            previousShape = userShape[:-len(newUserShape)]

            if len(gaps)>1:
                gap1 = gaps[-2]
                gap2 = gaps[-1]
                gap = np.sqrt((gap1[0]-gap2[0])*(gap1[0]-gap2[0]) + 0*(gap1[1]-gap2[1])*(gap1[1]-gap2[1]))

                if len(previousShape)>4:
                    #userShape = previousShape
                    letter_len, word = self.read(previousShape)

                    if gap>letter_len:
                        words+=word
                        words+=[' ']
                        #print 'SPAAAAAACE !!!!!!'
                        userShape = newUserShape


            if len(gaps)%2==0:
                gaps += [(touch.x, -touch.y)]

            #if  len(userShape)>4:
            #    letter_len = self.read(userShape)

            newUserShape = []
            #print len(userShape)


    def read(self,userShape):
        global words

        word = []
        total_score =0

        #scores = np.array(separator(userShape))
        numPointsInShape = len(userShape)/2

        complete = xyxy_to_xxyy(userShape)
        complete = np.reshape(complete,(-1,1))
        #complete = ShapeModeler.normaliseShapeHeight(np.array(complete))

        half_len_scan = max([10,int(len(complete)/14)])
        if half_len_scan%2>0:
            half_len_scan+=1
        #print half_len_scan


        scan = downsampleShape(userShape, half_len_scan, xyxyFormat=True)
        scan = np.reshape(scan, (-1, 1)) #explicitly make it 2D array with only one column
        scan = ShapeModeler.normaliseShapeWidth(np.array(scan))

        #print len(scan)

        if len(complete)<len(scan):
            complete=scan

        numPointsInScan = len(scan)/2

        scores = np.array(separator(scan))

        scores = scores/np.max(scores)
        scores = scores*np.abs(scores)
        scores[scores<0.1]=0

        indices = np.array(range(len(scores)))
        sep = indices[scores>0]
        #print zip(sep[:-1],sep[1:])

        for i in sep:
            x = scan[i]
            if len(scan[scan[i:numPointsInScan]<x])>0:
                scores[i]=0
            if len(scan[scan[:i]>x])>0:
                scores[i]=0

        scores[0]=1
        scores[-1]=1

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

        #plt.clf()
        #ShapeModeler.showShape_score(scan,scores)
        #ShapeModeler.showShape(complete)

        word = []
        word_score=[]

        #print len(complete)/len(scan)

        sep = indices[scores>0]
        for i,j in zip(sep[:-1],sep[1:]):

            ii = int(float(i)/numPointsInScan*numPointsInShape)
            jj = int(float(j)/numPointsInScan*numPointsInShape)+1

            #print ii
            #print jj

            # get the shape of the letter :
            shape = complete[ii:jj+1].T.tolist()[0] + complete[numPointsInShape+ii:numPointsInShape+jj+1].T.tolist()[0]
            shape = downsampleShape(shape, numPoints_shapeModeler)
            shape = np.reshape(shape,(-1,1))
            shape = ShapeModeler.normaliseShapeWidth(np.array(shape))

            best_letter = '?'
            errors = {}
            values = []
            for letter in abc:
                space = spaces[letter]
                error = space.getMinDist(shape)
                errors[letter] = error
                values.append(error)

            best_letter = min(errors, key = errors.get)
            error = errors[best_letter]
            word_score.append((best_letter,error))

            threshold =10

            #if best_letter in {'o','z'}:
                #threshold=10

            word.append(best_letter)

            '''if error<threshold:
                word.append(best_letter)
            else:
                scores[j] = 0 #or ii, best should to add with the letter (left/right) with worst error
            '''

        #plt.clf()
        #ShapeModeler.showShape_score(scan,scores)

        jump = False
        true_word = []
        for l1, l2 in zip(word[:-1],word[1:]):
            if not jump:

                if (l1=='r@' or l1=='v#' or l1=='s') and (l2=='r#' or l2=='z'):
                    true_word.append('r')
                    jump = True

                elif l1=='v@' and (l2=='r@' or l2=='v#'):
                    true_word.append('v')
                    jump=True

                elif (l1=='u' and (l2=='r@' or l2=='v#')) or (l1=='v@' and l2=='v'):
                    true_word.append('w')
                    jump=True

                elif (l1=='i' or l1=='v@') and (l2=='i' or l2=='v@'):
                    true_word.append('u')
                    jump=True

                elif (l1=='r@' or l1=='v#') and l2=='z':
                    true_word.append('r')
                    jump=True

                elif (l1=='i' or l1=='e' or l1=='r#' or l1=='z') and (l2=='r@' or l2=='v#'):
                    true_word.append('v')
                    jump=True

                elif (l1=='r@' or l1=='v#') and l2=='y':
                    true_word.append('r')
                    jump=True

                elif (l1=='l' or l1=='d' or l1=='h' or l1=='v#') and (l2=='r@' or l2=='v#'):
                    true_word.append('b')
                    jump=True

                elif l1=='l' and l2=='s':
                    true_word.append('b')
                    jump=True

                elif l1=='o' and l2=='on':
                    true_word.append('on')
                    jump=True

                elif l1=='o' and l2=='om':
                    true_word.append('om')
                    jump=True

                elif l1=='oi' and l2=='i':
                    true_word.append('o')
                    true_word.append('u')
                    jump=True

                elif (l1=='r@' or l1=='v#') and l2=='w':
                    true_word.append('r')
                    true_word.append('a')
                    jump=True

                elif l1=='r#':
                    true_word.append('r')

                elif l1=='u':
                    true_word.append('a')

                elif l1=='v':
                    true_word.append('o')

                else:
                    true_word.append(l1)
            else:
                jump = False
        if not jump:
            if word[-1]=='u':
                true_word.append('a')
            elif word[-1]=='v':
                true_word.append('o')
            else:
                true_word.append(word[-1])

        i=0
        for letter in true_word:
            if letter=='v@':
                # take the seconde score !!
                true_word[i] = 'i'
            if letter=='v#' or letter=='r@':
                true_word[i] = 'e'
            if letter=='r#':
                true_word[i] = 'r'
            i+=1



        #print 'found ' + str(true_word) 
        #print word_score

        #print '######################'
        #print ''
        #userShape = []
        #self.canvas.clear()

        pos1_x = float(complete[0])
        pos1_y = float(complete[numPointsInShape])
        pos2_x = float(complete[numPointsInShape-1])
        pos2_y = float(complete[-1])

        len_word = np.sqrt((pos1_x - pos2_x)*(pos1_x - pos2_x) + 0*(pos1_y - pos2_y) * (pos1_y - pos2_y))


        return len_word/float(len(word))*1.5, true_word

def separator(word):

    numPointsInWord = len(word)/2

    X = word[0:numPointsInWord]
    Y = word[numPointsInWord:]

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
