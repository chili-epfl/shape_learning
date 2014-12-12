#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from shape_learning.shape_modeler import ShapeModeler

import argparse
parser = argparse.ArgumentParser(description='Displays the effect of the parameters in a shape model');
parser.add_argument('shape', action="store",
                help='The shape to be visualised');
parser.add_argument('num_params', action="store", nargs='?', default=3, type=int,
                help='The number of parameters for which to have their effect visualised');
parser.add_argument('dataset_directory', action="store", nargs='?',
                help='The directory of the dataset to find the shape dataset');


#callback to modify shape based on parameter value changes
def update(val):
    global sliders , mainPlot, fig
    params = np.zeros((numParams,1));
    for i in range(numParams):
        params[i] = sliders[i].val;
    shape = shapeModeler.makeShape(params);
    shape = ShapeModeler.normaliseShape(shape);
    numPointsInShape = len(shape)/2;
    x_shape = shape[0:numPointsInShape];
    y_shape = shape[numPointsInShape:];
    mainPlot.set_data(x_shape, -y_shape);
    fig.canvas.draw_idle();


def reset(event):
    for i in range(numParams):
        sliders[i].reset();

def preparePlot(shapeModeler):
    global sliders, mainPlot, fig
    fig, ax = plt.subplots();
    whiteSpace = 0.15 + numParams*0.05;
    plt.subplots_adjust( bottom=whiteSpace);
    plt.axis('equal');
    
    #plot of initial shape
    params = np.zeros((numParams,1));
    shape = shapeModeler.makeShape(params);
    shape = ShapeModeler.normaliseShape(shape);
    numPointsInShape = len(shape)/2;
    x_shape = shape[0:numPointsInShape];
    y_shape = shape[numPointsInShape:];

    mainPlot, = plt.plot(x_shape, -y_shape);
    plt.axis([-1, 1, -1, 1],autoscale_on=False, aspect='equal');

    #add sliders to modify parameter values
    parameterVariances = shapeModeler.getParameterVariances();
    sliders = [0]*numParams;
    for i in range(numParams):
        slider = Slider(plt.axes([0.25, 0.1+0.05*(numParams-i-1), 0.65, 0.03], axisbg=axcolor),
             'Parameter '+str(i+1), -5*parameterVariances[i], 5*parameterVariances[i], valinit=0);
        slider.on_changed(update);
        sliders[i] = slider;
    
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
    button.on_clicked(reset)
    plt.show()
    

def prepareShapeModel(datasetDirectory, shape):
    import glob
    datasetFiles_shape = glob.glob(datasetDirectory + '/'+shape+'.dat')
    if(len(datasetFiles_shape)<1):
        raise Exception("Dataset not available at " + datasetDirectory + " for shape " + shape)
    shapeModeler = ShapeModeler(filename = datasetFiles_shape[0], num_principle_components = numParams)
    #import pdb; pdb.set_trace()
    return shapeModeler


shapeModeler = None;
axcolor = 'lightgoldenrodyellow';
sliders = None;
mainPlot = None;
fig = None;

if __name__ == "__main__":

    #parse arguments
    args = parser.parse_args();
    numParams = args.num_params;
    shape = args.shape;
    datasetDirectory = args.dataset_directory;
    
    #make shape model
    if(not datasetDirectory):
        import inspect
        fileName = inspect.getsourcefile(ShapeModeler);
        installDirectory = fileName.split('/lib')[0];
        datasetDirectory = installDirectory + '/share/shape_learning/letter_model_datasets/uji_pen_chars2';
    shapeModeler = prepareShapeModel(datasetDirectory, shape);

    #start gui
    preparePlot(shapeModeler)
    
    

