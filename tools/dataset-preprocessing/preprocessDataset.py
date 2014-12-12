import numpy
from scipy import interpolate
#from scipy.cluster.vq import vq, kmeans, whiten
from sklearn.cluster import MeanShift

MIN_CLUSTER_SIZE = 8
NB_POINTS=70

def interpolate_shape(shape,numDesiredPoints):
    """ Interpolate the shape to reach a predefined number of points, and
    switch from [xyxyxy...] to [xxx...yyy...]
    """
    numPointsInShape = len(shape)/2
    #make xyxy format
    x_shape = shape[0::2]
    y_shape = shape[1::2]

    t_current = numpy.linspace(0, 1, numPointsInShape)
    t_desired = numpy.linspace(0, 1, numDesiredPoints)

    f = interpolate.interp1d(t_current, x_shape, kind='cubic')
    x_shape = f(t_desired)

    f = interpolate.interp1d(t_current, y_shape, kind='cubic')
    y_shape = f(t_desired)

    shape = []
    shape[0:numDesiredPoints] = x_shape
    shape[numDesiredPoints:] = y_shape

    return shape

def normalize(shape):
    """Normalise shape so that max dimension is 1 
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
    if( scale<1e-10):
        print('Warning: shape is probably a bunch of points on top of each other... Skipping it.')
        return None

    x_shape = x_shape/scale
    y_shape = y_shape/scale

    newShape = numpy.zeros(len(shape))
    newShape[0:numPointsInShape] = x_shape
    newShape[numPointsInShape:] = y_shape
    return newShape

def cluster(samples):

    samples = numpy.array(samples)

    # clustering with k-Means
    #nb_clusters = 3
    #clusters, distortion = kmeans(whitened, nb_clusters)

    # clustering with mean-shift
    # http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html
    ms = MeanShift(bandwidth=1.90) # bandwidth manually estimated once with estimate_bandwidth
    ms.fit(samples)
    cluster_centers = ms.cluster_centers_
    labels = ms.labels_

    clusters = [list() for _ in range(len(cluster_centers))]

    for i, label in enumerate(labels):
        clusters[label].append(samples[i])
    
    return [cluster for cluster in clusters if len(cluster) >= MIN_CLUSTER_SIZE]

def preprocess(dataset):
    """ Pre-process the dataset to:
        - interpolate the sample so that they all have the same length
        - ensure a [xxxxx...yyyyyy...] layout
        - normalize the coordinates so that max dimension = 1
        - translate the shape so that the center is at (0,0)
        - returns it as a numpy array
        - cluster the samples into different allographs
    """
    sample_dict = {}
    clustered_dict = {}
    i = 0
    for char, samples in dataset.items():
        i += 1
        print("%.2f%% -- Interpolation and norlization of %d samples of <%s>..." % (float(i)/len(dataset) * 100, len(samples), char))

        #if char != "S": continue

        for sample in samples:
            if len(sample) > 1:
                # more than one stroke: ignore it for now
                continue

            shape = interpolate_shape(sample[0], NB_POINTS)
            norm_shape = normalize(shape)
            if norm_shape is not None:
                sample_dict.setdefault(char,[]).append(norm_shape)

    print("Done.\n\nClustering...")

    for k, v in sample_dict.items():
        clusters = cluster(v)
        print("Found %s clusters for <%s>" % (len(clusters), k))
        if len(clusters) in [0, 1]:
            clustered_dict[k] = v
        else:
            for i, c in enumerate(clusters):
                clustered_dict[k + str(i)] = c

    return clustered_dict




