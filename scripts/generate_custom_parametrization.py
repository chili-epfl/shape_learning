#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from shape_learning.shape_modeler import ShapeModeler
from functools import partial

import argparse

parser = argparse.ArgumentParser(description='Displays the effect of the parameters in a shape model')
parser.add_argument('symbols', action="store", 
                help='A regexp the symbol name should match to be processed. For instance, [a-z]')
parser.add_argument('num_params', action="store", nargs='?', default=3, type=int,
                help='The number of parameters for which to have their effect visualised')
parser.add_argument('dataset_directory', action="store", nargs='?',
                help='The directory of the dataset to find the shape dataset')
parser.add_argument('file', action="store",
                help='The file where the parametrization will be stored')


#callback to modify shape based on parameter value changes
def update(shapeModeler, val):
    global sliders , mainPlot, fig, pca_params
    pca_params = np.zeros((num_params, 1))

    for i in range(num_params):
        pca_params[i] = sliders[i].val
    
    shape = shapeModeler.makeShape(pca_params)
    shape = ShapeModeler.normaliseShape(shape)
    numPointsInShape = len(shape)/2
    x_shape = shape[0:numPointsInShape]
    y_shape = shape[numPointsInShape:]
    mainPlot.set_data(x_shape, -y_shape)
    fig.canvas.draw_idle()


def reset(event):
    for i in range(num_params):
        sliders[i].reset()

def param_gui(letter_name, shapeModeler):
    global sliders, mainPlot, fig, pca_params
    fig, ax = plt.subplots()
    whiteSpace = 0.15 + num_params*0.05
    plt.subplots_adjust( bottom=whiteSpace)
    plt.axis('equal')
    
    #plot of initial shape
    params = np.zeros((num_params,1))
    shape = shapeModeler.makeShape(params)
    shape = ShapeModeler.normaliseShape(shape)
    numPointsInShape = len(shape)/2
    x_shape = shape[0:numPointsInShape]
    y_shape = shape[numPointsInShape:]

    mainPlot, = plt.plot(x_shape, -y_shape)
    plt.axis([-1, 1, -1, 1],autoscale_on=False, aspect='equal')
    plt.title(letter_name)

    #add sliders to modify parameter values
    parameterVariances = shapeModeler.getParameterVariances()
    sliders = [0]*num_params
    for i in range(num_params):
        slider = Slider(plt.axes([0.25, 0.1+0.05*(num_params-i-1), 0.65, 0.03], axisbg=axcolor),
             'Parameter '+str(i+1), -5*parameterVariances[i], 5*parameterVariances[i], valinit=0)
        slider.on_changed(partial(update, shapeModeler))
        sliders[i] = slider
    
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
    button.on_clicked(reset)
    plt.show()

    return pca_params

def prepareShapesModel(datasetDirectory, regexp, num_params):

    import glob
    import os.path
    import re

    shapes = {}

    nameFilter = re.compile(regexp)
    
    datasets = glob.glob(datasetDirectory + '/*.dat')

    for dataset in datasets:
        name = os.path.splitext(os.path.basename(dataset))[0]

        if nameFilter.match(name):
            shapeModeler = ShapeModeler(filename = dataset, num_principle_components=num_params)
            shapes[name] = shapeModeler

    return shapes


pca_params = None
axcolor = 'lightgoldenrodyellow'
sliders = None
mainPlot = None
fig = None

if __name__ == "__main__":

    #parse arguments
    args = parser.parse_args()
    num_params = args.num_params
    datasetDirectory = args.dataset_directory
    regexp = args.symbols
    filename = args.file

    #make shape model
    if(not datasetDirectory):
        import inspect
        fileName = inspect.getsourcefile(ShapeModeler)
        installDirectory = fileName.split('/lib')[0]
        datasetDirectory = installDirectory + '/share/shape_learning/letter_model_datasets/uji_pen_chars2'
    shapes = prepareShapesModel(datasetDirectory, regexp, num_params)

    params = {}

    for k, v in shapes.items():
        pca_params = np.zeros((num_params, 1))
        params[k] = param_gui(k, v)

    with open(filename, 'w') as f:
        f.write("#Principle components values for a %s-dimensional PCA\n" % num_params)
        for k, v in params.items():
            f.write("[" + k + "]\n")
            f.write("%s\n" % " ".join([str(i[0]) for i in v]))
