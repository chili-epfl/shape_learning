# -*- coding: utf-8 -*-
"""
This script generates different kind of graphs outputs using reference shapes
and demo shapes recovered from log files

@author: ferran
"""

import string
import datetime
import glob
import os.path
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
import inspect
from shape_learning.shape_modeler import ShapeModeler

"""
Recovers the specified demo shapes provided by the children 
and stored in all log files based on specific period of time
""" 
def getLogShapes(logDir, initDate, endDate, leterList, num_params):
    
    initDate = datetime.datetime.strptime(initDate, "%d-%m-%Y")
    endDate = datetime.datetime.strptime(endDate, "%d-%m-%Y")        
    
    logset = glob.glob(logDir + '/*.log')
    logDate = []
    for i in range(len(logset)):
        logDate.append(find_between(logset[i], "es-", ".log"))
    
    dates = [datetime.datetime.strptime(ts, "%d-%m-%Y") for ts in logDate]
    dates.sort()
    sortList = [datetime.datetime.strftime(ts, "%d-%m-%Y") for ts in dates]
    alphabet = re.compile(regexp)

    fields = {key: [] for key in leterList}
    
    #Prepare the counter structure to keep where appears each shape occurence
    sizeLog = len(logset)
    count = {key: np.zeros(sizeLog) for key in leterList}

    ind = 0
    for fileDate in sortList:
        logfile = logDir + "/shapes-" + fileDate + ".log"
        #print('Evaluating date ... ' + logfile)
        date = re.search("([0-9]{2}\-[0-9]{2}\-[0-9]{4})", logfile)
        logDate = datetime.datetime.strptime(date.group(1), "%d-%m-%Y")
              
        if (logDate >= initDate and logDate <= endDate): 
            #print('Opening ... ' + logfile)                  
            #Process file
            with open('%s' %logfile, 'r') as inF:
                lineNumb = 1
                for line in inF:
                    found = re.search(r'...new\b.*?\bdemonstration\b', line)
                    if found is not None:                    
                        leter = found.group(0)[0]
                        if alphabet.match(leter):
                            #print('Match found! ' + leter + ' in line ' + str(lineNumb))
                            
                            #Keep the sum of each leter occurrence                           
                            count[leter][ind] = count[leter][ind] + 1

                            strShape = line.partition("Path: [")[-1].rpartition(']')[0]                           
                            shape = strShape.split(', ')
                            shape = np.array(map(float, shape))
                            #The points need to be reshaped to have format (140,1)
                            shape = np.reshape(shape, (-1, 1))

                            fields[leter].append(shape)

                    lineNumb = lineNumb + 1
        else:
            raise Exception("No log files among the entry dates")
        ind = ind +1
  
    return fields, count


"""
Returns the content between two strings
"""
def find_between(s, first, last):
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return s[start:end]
    except ValueError:
        return ""
        
               
"""
Returns a list of ShapeModeler objects based on a regexp
"""         
def prepareShapesModel(datasetDirectory, num_params):

    shapes = {}

    nameFilter = re.compile(regexp)    
    datasets = glob.glob(datasetDirectory + '/*.dat')
    
    for dataset in datasets:
        name = os.path.splitext(os.path.basename(dataset))[0]

        if nameFilter.match(name):
            shapeModeler = ShapeModeler(init_filename = dataset, num_principle_components=num_params)
            shapes[name] = shapeModeler
           
    return shapes


"""
Returns a dictionary composed by the Key {a,b,c,d...} and its reference shape
"""
def getRefShapes(shapesModel, leterList):

    refShapes = {key: [] for key in leterList}
    
    for leter in shapesModel:
        refShape = np.reshape(shapesModel[leter].dataMat[0], (-1, 1))
        refShapes[leter].append(refShape)
    
    return refShapes      


"""
Projects a set of shapes into the correspondent eigenspace
"""
def projectShape(shapesModel, shapes, num_params, leterList):
    
    projShapes = {key: [] for key in leterList}

    for leter in shapesModel:
        for shape in shapes:
            if leter == shape:
                i=0
                for sample in range(len(shapes[shape])):                
                    projShape = shapesModel[leter].decomposeShape(shapes[leter][sample])
                    i = i+1
                    projShapes[leter].append(projShape[0][0:])
                #print '%s: is present in both with %i samples' % (leter,i)
    
    return projShapes

      
"""
Calculates the distance between the demo shapes and its references
returned in a date/shape sorted structure
"""
def calcDistance(projRefShapes, projDemoShapes, leterList):
    
    distances = {key: [] for key in leterList}
    
    for leter in projRefShapes:
        refLeter = projRefShapes[leter]
        for i in range(len(projDemoShapes[leter])):
            demoLeter = projDemoShapes[leter][i]
            dist = np.linalg.norm(demoLeter-refLeter)
            distances[leter].append(dist)        
        
    return distances


"""
Returns the list leters based on the regular expression
"""
def getLeterList():
    
    alphabetList = string.ascii_lowercase   
    alphabet = re.compile(regexp)
    leterList = re.findall(alphabet, alphabetList)
    
    return leterList
    
       
"""
Receives the 'dist' structure and plots for each type of shape, a graph
"""
def DistStaticEigen(datasetDirectory, logDir, num_params, initDate, endDate):
    
    leterList = getLeterList()
        
    #Prepare the Eigenspaces for each shape and store them
    shapesModel = prepareShapesModel(datasetDirectory, num_params)    
    
    #Obtain the reference shapes and project them into the correspondent eigenspace    
    refShapes = getRefShapes(shapesModel, leterList)
    projRefShapes = projectShape(shapesModel, refShapes, num_params, leterList)
    
    #Obtain the demo shapes from the logs and project them into the eigenspace
    demoShapes = getLogShapes(logDir, initDate, endDate, leterList, num_params)
    projDemoShapes = projectShape(shapesModel, demoShapes[0], num_params, leterList)
    
    #Calculate the distance between the same type of shape
    distances = calcDistance(projRefShapes, projDemoShapes, leterList)         
       
    return distances
    

"""
Receives the 'dist' structure and plots for each type of shape, a graph
taking into account the dynamics of the eigenspace
"""   
def DistDynEigen(datasetDirectory, logDir, num_params, initDate, endDate):
    
    leterList = getLeterList()       
    distances = {key: [] for key in leterList}
       
    #Prepare the INITIAL Eigenspaces for each shape and store them
    shapesModel = prepareShapesModel(datasetDirectory, num_params)
    
    #Obtain all demo leters
    demoShapes = getLogShapes(logDir, initDate, endDate, leterList, num_params)
    demo = demoShapes[0]
    separator = demoShapes[1]
    
    #Obtain all reference leters
    refShapes = getRefShapes(shapesModel, leterList)
    
    for leter in demo:
        numShapes = len(demo[leter])
        print '%i occurrences for letter %s' %(numShapes,leter)
       
        for i in range(numShapes):
            #Can be the case there is not dataset for a leter, in this case
            # the reference shape for that leter will be empty
            if len(refShapes[leter])>0:
                #Project the demo leter
                demoShape = {leter: []}
                demoShape[leter].append(demo[leter][i])
                projDemoShape = projectShape(shapesModel, demoShape, num_params, leterList)
        
                #Project the reference leter
                refShape = {leter: []}
                refShape[leter].append(refShapes[leter][0])
                projRefShape = projectShape(shapesModel, refShape, num_params, leterList)
            
                #Calculate the distance between projections
                singleDist = calcDistance(projRefShape, projDemoShape, leterList)
                distances[leter].append(singleDist[leter][0])            
                
                #Realculate the eigenspace using the demo leter
                shapesModel[leter].extendDataMat(demo[leter][i])
            
    return distances, separator

    
"""
Generates as much plots as shapes the regexp indicates taking into account a
dynamic and static eigenspace
"""    
def plotDistances():
    
    leterList = getLeterList()
    
    #Calculations to locate the subplots in the best way    
    numLeter=len(leterList)
    col = np.floor(numLeter**0.5).astype(int)
    row = np.ceil(1.*numLeter/col).astype(int)
    print "col\t=\t%d\nrow\t=\t%d\ncol*row\t=\t%d\ntotal\t=\t%d" % (col,row,col*row,numLeter)
    fig = plt.figure(1, figsize=(2.*col,2.*row))
    subplots_adjust(hspace=0.000)
    
    #Get the distances for Dynamic and Static Eigenspace
    distDynEigen = DistDynEigen(datasetDirectory, logDir, num_params, initDate, endDate) 
    distStaticEigen = DistStaticEigen(datasetDirectory, logDir, num_params, initDate, endDate)
    
    #For the vertical red lines
    separator = distDynEigen[1]
    distDynEigen = distDynEigen[0]
    
    for i in range(1,numLeter+1):
        ax = fig.add_subplot(row,col,i)       
        #Make the x-axis start at unit 1 for better understanding
        ax.plot(range(1,len(distStaticEigen[leterList[i-1]])+1), distStaticEigen[leterList[i-1]], '-bo',
                range(1,len(distDynEigen[leterList[i-1]])+1), distDynEigen[leterList[i-1]], '-go')
        ax.set_title('Shape: ' + leterList[i-1])
        plt.xlabel('Time progression')
        plt.ylabel('dist. wrt. Ref')
        plt.ylim([0,7])
        
        numLogs = len(separator[leterList[i-1]])
        sumat = 0
        
        #The -1 avoids to put the last line               
        for j in range(numLogs-1):
            vline = separator[leterList[i-1]][j]+sumat            
            #Paint the line if it is not at 0 or outside range
            if vline > 0 and vline < len(distDynEigen[leterList[i-1]]):
                axvline(vline+0.5, color='r')            
            sumat = separator[leterList[i-1]][j]
            
    fig.set_tight_layout(True)
                 
    return fig

   
"""
Plots the all occurrences of a letter in the 3D space defined by the tree most 
important eigenvalues
"""
def plot3D():
    
    leterList = getLeterList()
        
    #Prepare the Eigenspaces for each shape and store them
    shapesModel = prepareShapesModel(datasetDirectory, num_params)
    
    #Obtain the demo shapes from the logs and project them into the eigenspace
    demoShapes = getLogShapes(logDir, initDate, endDate, leterList, num_params)
    projDemoShapes = projectShape(shapesModel, demoShapes[0], num_params, leterList)
    
    #Letter to plot the eigenspace and its instances projected on it  
    fig = plt.figure(2)
    ax = fig.gca(projection='3d')
    ax.set_title('Eigenspace of shape: ' + letter)
    
    for i in range(len(projDemoShapes[letter])):
        demoLeter = projDemoShapes[letter][i]
        ax.scatter(demoLeter[0], demoLeter[1], demoLeter[2])
    
    #Reference letter representing a high value in each of the important parameters
    modelx = np.array([2.44,-0.08,-0.5])
    modely = np.array([0.17,1.3,-0.06])
    modelz = np.array([-2.1,-0.03,2.38])
    
    ax.scatter(modelx[0], modelx[1], modelx[2], c='r')
    ax.scatter(modely[0], modely[1], modely[2], c='r')
    ax.scatter(modelz[0], modelz[1], modelz[2], c='r')
    
    #Define names and limits
    ax.set_xlabel('X axis - F1')
    ax.set_ylabel('Y axis - F2')
    ax.set_zlabel('Z axis - F3')
    ax.set_xlim(-2.5,2.5)
    ax.set_ylim(-2.3,1.3)
    ax.set_zlim(-2.15,2.40)
    
    return fig


#Values to modify if it is the case
letter = 'a'  
regexp = '[a-z]'
fileName = inspect.getsourcefile(ShapeModeler);
installDirectory = fileName.split('/lib')[0];    
initDate = '01-01-2000';
endDate = '01-01-2100';
num_params = 3   
datasetDirectory = installDirectory + '/share/shape_learning/letter_model_datasets/alexis_set_for_children'; #uji_pen_subset
logDir = '../logs';

plotDistances()
plot3D()
plt.show()