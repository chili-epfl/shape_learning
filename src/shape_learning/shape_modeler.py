""" Class to decompose a dataset of shapes into its principle components,
and to make and show new shapes which are represented by the mean shape
plus some amount of said principle components.
"""

import numpy
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA
import random
from copy import deepcopy

class ShapeModeler:

    def __init__(self, shape_name=None, samples=None, filename=None, num_principle_components=10):
        """ Initialize a shape modeler

        If given a dataset (params samples or filename), loads the training
        dataset for a given shape, and run a PCA decomposition on it.

        :param samples: a list of sample shapes, each presented as a list of coordinates [x1,x2,...,y1,y2...].
                        Each shape must have the same number of coordinates.
        :param filename: the path to a shape dataset. See makeDataMatrix for the expected format.
        :paran num_principle_components: the number of desired principle components
        """

        self.shape_name = shape_name
        self.num_principle_components = num_principle_components

        if samples is None and filename is None:
            return

        if samples:
            self.dataMat = numpy.array(samples)

        elif filename:
            self.makeDataMatrix(filename)

        self.performPCA()

    def makeDataMatrix(self, filename):
        """Read data from text file and store resulting data matrix

        For n samples of m points, the text file should be formatted as:

            n
            m
            x11 x12 ... x1m y11 y12 ... y1m
            x21 x22 ... x2m y21 y22 ... y2m
            ...
            xn1 xn2 ... xnm yn1 yn2 ... ynm

        """
        with open(filename) as f:
            self.numShapesInDataset = int(f.readline().strip())
            self.numPointsInShapes = int(f.readline().strip())
            if not(self.numShapesInDataset and self.numPointsInShapes):
                raise RuntimeError("Unable to read sizes needed from text file")

            self.dataMat = numpy.empty((self.numShapesInDataset, self.numPointsInShapes*2))
            for i in range(self.numShapesInDataset):
                line = f.readline().strip()
                values = line.split(' ')
                if not(len(values) == self.numPointsInShapes*2):
                    raise RuntimeError("Unable to read appropriate number of points from text file for shape "+str(i+1))

                self.dataMat[i] = map(float, values)

    def performPCA(self):
        """ Calculate the top 'num_principle_components' principle components of
        the dataset, the observed variance of each component, and the mean
        """
        covarMat = numpy.cov(self.dataMat.T)
        eigVals, eigVecs = numpy.linalg.eig(covarMat)
        self.principleComponents = numpy.real(eigVecs[:, 0:self.num_principle_components])
        self.parameterVariances = numpy.real(eigVals[0:self.num_principle_components])
        self.meanShape = self.dataMat.mean(0).reshape((self.numPointsInShapes*2, 1))

    def getParameterVariances(self):
        """ Return the variances associated which each of the top principle components
        """
        return self.parameterVariances

    def makeShape(self, params):
        """ Generate a shape with the given parameter vector
        """
        if(not params.shape == (self.num_principle_components, 1)):
            raise RuntimeError("Vector of parameters must have dimensions of (num_principle_components,1)")
        shape = self.meanShape + numpy.dot(self.principleComponents, params)
        return shape

    def makeShapeVaryingParam(self, paramsToVary, paramValues):
        """ Generate a shape modifying the given parameter
        """
        xb = numpy.zeros((self.num_principle_components, 1))
        for i in range(len(paramsToVary)):
            xb[paramsToVary[i]-1, 0] = paramValues[i]
        shape = self.makeShape(xb)
        return shape, xb

    def makeRandomShapeFromUniform(self, params, paramsToVary, bounds):
        """ Draw 'paramsToVary' values from uniform distribution with limits
        given by 'bounds' and make shape
        """
        xb = deepcopy(params)
        for i in range(len(paramsToVary)):
            sample = random.uniform(bounds[i, 0], bounds[i, 1])
            xb[paramsToVary[i]-1, 0] = sample
        shape = self.makeShape(xb)
        return shape, xb

    def makeRandomShapeFromTriangular(self, params, paramsToVary, bounds, modes):
        """ Draw 'paramsToVary' values from triangular distribution with limits
        given by 'bounds' and modes given by 'modes' and make shape       
        """
        b = deepcopy(params)
        for i in range(len(paramsToVary)):
            sample = random.triangular(bounds[i, 0], modes[i], bounds[i, 1])
            b[paramsToVary[i]-1, 0] = sample
        return self.makeShape(b), b

    def decomposeShape(self, shape):
        """ Convert shape into its 'num_principle_components' parameter values
        (project it onto the num_principle_components-dimensional space)
        """
        if(not shape.shape == (self.numPointsInShapes*2, 1)):
            raise RuntimeError("Shape to decompose must be the same size as shapes used to make the dataset")
        params = numpy.dot(self.principleComponents.T, shape - self.meanShape)

        approxShape = self.meanShape + numpy.dot(self.principleComponents, params)
        diff = abs(shape-approxShape)**2
        error = sum(diff)/(self.numPointsInShapes*2)
        return params, error

    @staticmethod
    def showShape(shape):
        """ Show shape with random colour
        """
        numPointsInShape = len(shape)/2
        x_shape = shape[0:numPointsInShape]
        y_shape = shape[numPointsInShape:]

        plt.plot(x_shape, -y_shape, c=numpy.random.rand(3,1))
        plt.axis([-1, 1, -1, 1])
        plt.draw();#show(block=False)

    @staticmethod
    def normaliseShape(shape):
        """ Normalise shape so that max dimension is 1 
        """
        numPointsInShape = len(shape)/2
        x_shape = shape[0:numPointsInShape]
        y_shape = shape[numPointsInShape:]

        #shift so centre of shape is at (0,0)
        x_range = max(x_shape)-min(x_shape)
        y_range = max(y_shape)-min(y_shape)
        x_shape = x_shape-(max(x_shape)-x_range/2)
        y_shape = y_shape-(max(y_shape)-y_range/2)

        #normalise shape
        scale = max(x_range,y_range)
        if scale < 1e-10:
            print('Warning: shape is probably a bunch of points on top of each other...')

        x_shape = x_shape/scale
        y_shape = y_shape/scale

        newShape = numpy.zeros(shape.shape)
        newShape[0:numPointsInShape] = x_shape
        newShape[numPointsInShape:] = y_shape
        return newShape

    @staticmethod
    def getShapeCentre(shape):
        """ Calculate the centre of the shape
        """
        numPointsInShape = len(shape)/2
        x_shape = shape[0:numPointsInShape]
        y_shape = shape[numPointsInShape:]

        x_range = max(x_shape)-min(x_shape)
        y_range = max(y_shape)-min(y_shape)
        x_centre = (max(x_shape)-x_range/2)
        y_centre = (max(y_shape)-y_range/2)
        return [x_centre, -y_centre]

    @staticmethod
    def normaliseShapeHeight(shape):
        """ Normalise shape so that height is 1 
        """
        numPointsInShape = len(shape)/2
        x_shape = shape[0:numPointsInShape]
        y_shape = shape[numPointsInShape:]

        #shift so centre of shape is at (0,0)
        x_range = max(x_shape)-min(x_shape)
        y_range = max(y_shape)-min(y_shape)
        x_centre = (max(x_shape)-x_range/2)
        y_centre = (max(y_shape)-y_range/2)
        x_shape = x_shape-x_centre
        y_shape = y_shape-y_centre

        #normalise shape
        scale = y_range
        if scale < 1e-10:
            print('Warning: shape is probably a bunch of points on top of each other...')

        x_shape = x_shape/scale
        y_shape = y_shape/scale

        newShape = numpy.zeros(shape.shape)
        newShape[0:numPointsInShape] = x_shape
        newShape[numPointsInShape:] = y_shape
        return newShape

    @staticmethod
    def normaliseAndShowShape(shape):
        """ Normalise shape so that max dimension is 1 and then show
        """
        shape = ShapeModeler.normaliseShape(shape)
        ShapeModeler.showShape(shape)

    def normaliseMeanShapeHeight(self):
        self.meanShape = ShapeModeler.normaliseShapeHeight(self.meanShape)
