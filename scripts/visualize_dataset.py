#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from shape_learning.shape_modeler import ShapeModeler

import argparse


parser = argparse.ArgumentParser(description='Displays all the characters in the given dataset')
parser.add_argument('shapes', action="store", nargs='?', default = "[a-z]",
                help='A regexp the shapes name should match to be visualised')
parser.add_argument('--dataset_directory', action="store", nargs='?',
                help='The directory of the dataset to find the shape dataset')
parser.add_argument('--parameters', action="store", nargs='?',
                help='A predefined set of principle parameters to use')


axcolor = 'lightgoldenrodyellow'
sliders = None
mainPlot = None
fig = None



def prepareShapesModel(datasetDirectory, regexp=".*"):

    import glob
    import os.path
    import re

    shapes = {}

    nameFilter = re.compile(regexp)
    
    datasets = glob.glob(datasetDirectory + '/*.dat')

    for dataset in datasets:
        name = os.path.splitext(os.path.basename(dataset))[0]

        if nameFilter.match(name):
            shapeModeler = ShapeModeler(filename = dataset, num_principle_components = 3)
            shapes[name] = shapeModeler

    return shapes

def parse_parameters(filename):

    params = {}
    with open(filename) as f:

        key = None
        for l in f.readlines():
            if l.startswith("#") or l.rstrip()=="": continue
            if l.startswith("["): key = l[1:-2]
            else: params[key] = np.array([[float(p)] for p in l.split()])

    return params

if __name__ == "__main__":

    #parse arguments
    args = parser.parse_args()
    datasetDirectory = args.dataset_directory
    regexp = args.shapes
    
    initial_params = {}
    if args.parameters:
        initial_params = parse_parameters(args.parameters)
    print(initial_params)

    if(not datasetDirectory):
        import inspect
        fileName = inspect.getsourcefile(ShapeModeler)
        installDirectory = fileName.split('/lib')[0]
        datasetDirectory = installDirectory + '/share/shape_learning/letter_model_datasets/uji_pen_chars2'

    shapes = prepareShapesModel(datasetDirectory, regexp)

    print("I will display the following shapes:\n%s" % " ".join(shapes.keys()))

    for n, k in enumerate(sorted(shapes.keys())):

        shape = shapes[k].meanShape

        plt.subplot(5, len(shapes)/5 + 1, n+1)

        numPointsInShape = len(shape)/2
        x_shape = shape[0:numPointsInShape]
        y_shape = shape[numPointsInShape:]

        plt.plot(x_shape, -y_shape, c='k')
        
        if k in initial_params:
            shape = shapes[k].makeShape(initial_params[k])
            x_shape = shape[0:numPointsInShape]
            y_shape = shape[numPointsInShape:]

            plt.plot(x_shape, -y_shape, c='r')
            
        plt.title(k)
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
    plt.show()


