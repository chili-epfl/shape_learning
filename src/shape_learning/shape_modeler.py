"""
Class to decompose a dataset of shapes into its principle components, 
and to make and show new shapes which are represented by the mean shape
plus some amount of said principle components.
"""

import numpy
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA
import random
from copy import deepcopy

class ShapeModeler:
    
    #Read data from text file and store resulting data matrix
    #Text file should be formatted as:
    #numShapesInDataset \n numPointsInShapes \n 
    #shape_1_x_1 shape_1_x_2 ... shape_1_x_numPointsInShapes shape_1_y_1
    # shape_1_y_2 ... shape_1_y_numPointsInShapes \n
    #shape_2_x_1 shape_2_x_2 ... shape_2_x_numPointsInShapes shape_2_y_1
    # shape_2_y_2 ... shape_2_y_numPointsInShapes \n
    # ...
    #shape_numShapesInDataset_x_1 shape_numShapesInDataset_x_2 ...
    # shape_numShapesInDataset_x_numPointsInShapes 
    #shape_numShapesInDataset_y_1 shape_numShapesInDataset_y_2 ...
    # shape_numShapesInDataset_y_numPointsInShapes
    def makeDataMatrix(self, filename):
        f = open(filename);
        self.numShapesInDataset = int(f.readline().strip());
        self.numPointsInShapes = int(f.readline().strip());
        if(not (self.numShapesInDataset and self.numPointsInShapes) ):
            raise RuntimeError("Unable to read sizes needed from text file");
            
        
        self.dataMat = numpy.empty((self.numShapesInDataset, self.numPointsInShapes*2));
        for i in range(self.numShapesInDataset):
            line = f.readline().strip();
            values = line.split(' ');
            if(not (len(values) == self.numPointsInShapes*2) ):
                raise RuntimeError("Unable to read appropriate number of points from text file for shape "+str(i+1));  
                        
            self.dataMat[i] = map(float,values); 
        f.close();    
        
    #Calculate the top 'numPrincipleComponents' principle components of 
    #the dataset, the observed variance of each component, and the mean
    def performPCA(self, numPrincipleComponents):
        self.numPrincipleComponents = numPrincipleComponents;
        covarMat = numpy.cov(self.dataMat.T);
        eigVals, eigVecs = numpy.linalg.eig(covarMat);
        self.principleComponents = numpy.real(eigVecs[:,0:self.numPrincipleComponents]);
        self.parameterVariances = numpy.real(eigVals[0:self.numPrincipleComponents]);
        self.meanShape = self.dataMat.mean(0).reshape((self.numPointsInShapes*2,1));    
    
    #Return the variances associated which each of the top principle components
    def getParameterVariances(self):
        return self.parameterVariances;
     
    #Generate a shape with the given parameter vector  
    def makeShape(self, params):
        if(not params.shape == (self.numPrincipleComponents,1)):
            raise RuntimeError("Vector of parameters must have dimensions of (numPrincipleComponents,1)")
        shape = self.meanShape + numpy.dot(self.principleComponents,params);
        return shape;  
              
    #Generate a shape modifying the given parameter
    def makeShapeVaryingParam(self, paramsToVary, paramValues):
        xb = numpy.zeros((self.numPrincipleComponents,1));
        for i in range(len(paramsToVary)):
            xb[paramsToVary[i]-1,0] = paramValues[i];
        shape = self.makeShape(xb);
        return shape, xb;
        
    #Draw 'paramsToVary' values from uniform distribution with limits given by 'bounds' and make shape
    def makeRandomShapeFromUniform(self, params, paramsToVary, bounds):
        xb = deepcopy(params);
        for i in range(len(paramsToVary)):
            sample = random.uniform(bounds[i,0],bounds[i,1]);
            xb[paramsToVary[i]-1,0] = sample;
        shape = self.makeShape(xb);
        return shape, xb;    
        
    #Draw 'paramsToVary' values from triangular distribution with limits given by 'bounds' and modes given by 'modes' and make shape       
    def makeRandomShapeFromTriangular(self, params, paramsToVary, bounds, modes):
        b = deepcopy(params);
        for i in range(len(paramsToVary)):
            sample = random.triangular(bounds[i,0],modes[i],bounds[i,1]);
            b[paramsToVary[i]-1,0] = sample;
        return self.makeShape(b), b;  
        
    #Convert shape into its 'numPrincipleComponents' parameter values (project it onto the numPrincipleComponents-dimensional space)
    def decomposeShape(self, shape):
        if(not shape.shape == (self.numPointsInShapes*2,1)):
            raise RuntimeError("Shape to decompose must be the same size as shapes used to make the dataset");
        params = numpy.dot(self.principleComponents.T, shape - self.meanShape);

        approxShape = self.meanShape + numpy.dot(self.principleComponents, params);
        diff = abs(shape-approxShape)**2;
        error = sum(diff)/(self.numPointsInShapes*2);
        return params, error
        
    #Show shape with random colour
    @staticmethod
    def showShape(shape):
        numPointsInShape = len(shape)/2;
        x_shape = shape[0:numPointsInShape];
        y_shape = shape[numPointsInShape:];

        plt.plot(x_shape, -y_shape, c=numpy.random.rand(3,1))
        plt.axis([-1,1,-1,1]);
        plt.draw();#show(block=False);
    
    #Normalise shape so that max dimension is 1 
    @staticmethod
    def normaliseShape(shape):
        numPointsInShape = len(shape)/2;
        x_shape = shape[0:numPointsInShape];
        y_shape = shape[numPointsInShape:];
        
        #shift so centre of shape is at (0,0)
        x_range = max(x_shape)-min(x_shape);
        y_range = max(y_shape)-min(y_shape);
        x_shape = x_shape-(max(x_shape)-x_range/2);
        y_shape = y_shape-(max(y_shape)-y_range/2);
        
        #normalise shape
        scale = max(x_range,y_range);
        if( scale<1e-10):
            print('Warning: shape is probably a bunch of points on top of each other...')
        
        x_shape = x_shape/scale;
        y_shape = y_shape/scale;        
        
        newShape = numpy.zeros(shape.shape);
        newShape[0:numPointsInShape] = x_shape;
        newShape[numPointsInShape:] = y_shape;
        return newShape
        
    #Calculate the centre of the shape
    @staticmethod
    def getShapeCentre(shape):
        numPointsInShape = len(shape)/2;
        x_shape = shape[0:numPointsInShape];
        y_shape = shape[numPointsInShape:];

        x_range = max(x_shape)-min(x_shape);
        y_range = max(y_shape)-min(y_shape);
        x_centre = (max(x_shape)-x_range/2);
        y_centre = (max(y_shape)-y_range/2);
        return [x_centre,-y_centre];
        
    #Normalise shape so that height is 1 
    @staticmethod
    def normaliseShapeHeight(shape):
        numPointsInShape = len(shape)/2;
        x_shape = shape[0:numPointsInShape];
        y_shape = shape[numPointsInShape:];
        
        #shift so centre of shape is at (0,0)
        x_range = max(x_shape)-min(x_shape);
        y_range = max(y_shape)-min(y_shape);
        x_centre = (max(x_shape)-x_range/2);
        y_centre = (max(y_shape)-y_range/2);
        x_shape = x_shape-x_centre; 
        y_shape = y_shape-y_centre;
        
        #normalise shape
        scale = y_range;
        if( scale<1e-10):
            print('Warning: shape is probably a bunch of points on top of each other...')
        
        x_shape = x_shape/scale;
        y_shape = y_shape/scale;
        
        newShape = numpy.zeros(shape.shape);
        newShape[0:numPointsInShape] = x_shape;
        newShape[numPointsInShape:] = y_shape;
        return newShape
        
    #Normalise shape so that max dimension is 1 and then show   
    @staticmethod
    def normaliseAndShowShape(shape):
        shape = ShapeModeler.normaliseShape(shape);
        ShapeModeler.showShape(shape);
            
    def normaliseMeanShapeHeight(self):
        self.meanShape = ShapeModeler.normaliseShapeHeight(self.meanShape);
