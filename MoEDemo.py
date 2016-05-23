from MixtureOfExperts import *
from optparse import OptionParser
import numpy as np
import random

##########################################
##########  TEST    DATA    ##############
#########################################
def function1(data):
    out = list()
    for val in data:
        offset = np.random.normal(loc=0.0, scale=1.0)
        out.append( ((2 * val)) + offset-5)
    return out

def function2(data):
    out = list()
    for val in data:
        offset = np.random.normal(loc=0.0, scale=1.0)
        out.append( ((-2 * (val-5) + 5) + offset) )
    return out

def function3(data):
    out = list()
    for val in data:
        out.append( 3*(val**2) + 5*val + 2 )
    return out

def createData():
    data1 = list(); data2 = list()
    [data1.append(random.uniform(0, 5)) for i in range(100)]
    [data2.append(random.uniform(5,10)) for i in range(100)]

    data = list()
    data.extend(data1); data.extend(data2)

    out1 = function3(data1); out2 = function3(data2)

    out = list()
    out.extend(out1); out.extend(out2)
    
    #function3(data)
    return np.asarray(data), np.asarray(out)

def createData2():
    data1 = list(); data2 = list()
    [data1.append(random.uniform(0, 5)) for i in range(100)]
    [data2.append(random.uniform(5,10)) for i in range(100)]

    data = list()
    data.extend(data1); data.extend(data2)

    out1 = function3(data); out2 = function2(data)
    out = list()
    for i in range(len(data)):
        out.append( (out1[i], out2[i]) )

    #function3(data)
    return np.asarray(data), np.asarray(out)


def function2D(x, y):
    out = list()
    for i in range(len(x)):
        zx = -2*(x[i]-5)**2 + 50
        zy = -2*(y[i]-5)**2 + 50

        out.append( math.log(zx + zy) )
    return out

def function2D2(x, y):
    out = list()
    for i in range(len(x)):
        zx = 2*(x[i]-5)**2 + 50
        zy = 2*(y[i]-5)**2 + 50

        out.append( math.log(zx + zy) )
    return out


def createData2D():
    data1, data2 = [], []
    [data1.append(random.uniform(0,10)) for i in range(200)]
    [data2.append(random.uniform(0,10)) for i in range(200)]

    data = list()
    for i in range(len(data1)):
        data.append( (data1[i], data2[i]) )

    out = function2D(data1, data2)

    return np.asarray(data), np.asarray(out)


def createData2D2():
    data1, data2 = [], []
    [data1.append(random.uniform(0,10)) for i in range(200)]
    [data2.append(random.uniform(0,10)) for i in range(200)]

    data = list()
    for i in range(len(data1)):
        data.append( (data1[i], data2[i]) )

    out1 = function2D(data1, data2)
    out2 = function2D2(data1, data2)

    out = list()
    for i in range(len(data1)):
        out.append( (out1[i], out2[i]) )


    return np.asarray(data), np.asarray(out)




def readFile(filename):
    f = open(filename, "r")
    lines = f.readlines()
    data = list(); out = list()

    for line in lines[1:]:
        line = line.rstrip('\n')
        split = line.split(';')
        split_data = [float(x) for x in split[0].split(",")]
        split_output = [float(x) for x in split[1].split(",")]

        if len(split_data) > 1:
            data.append( tuple(split_data) )
        elif len(split_data) == 1:
            data.append(split_data[0])

        if len(split_output) > 1:
            out.append( tuple(split_output) )
        elif len(split_output) == 1:
            out.append(split_output[0])

    f.close()
    return np.asarray(data), np.asarray(out)

################################################



def function2Dcompare(X, Y):
    return np.log((-2*(X-5)**2 + 50) + (-2*(Y-5)**2 + 50))

def functionPeaksCompare(X, Y):
    return 3*(1-X)**2*np.exp(-X**2 - (Y+1)**2) - 10*(X/5 - X**3 - Y**5) * np.exp(-X**2 - Y**2) - (1/3)*np.exp(-(X+1)**2 - Y**2)

if __name__ == '__main__':
    #Options to run from command line
    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option("-n", "--num", action="store", help="number of experts", type="int")
    parser.add_option("-i", "--iter", action="store", help="maximum iterations", type="int")
    parser.add_option("-t", "--train", action="store", help="train set filename (none for synthetic data)")
    parser.add_option("-v", "--test", action="store", help="test set filename (none for synthetic data)")

    (options, args) = parser.parse_args()

    data, out = createData2D2() if options.train is None else readFile(options.train)

    num_experts = 2 if options.num is None else options.num
    maxIterations = 100 if options.iter is None else options.iter


    #Initialize and format training data
    training_y = out[30:170:1] if options.train is None else out
    training_x = data[30:170:1] if options.train is None else data

    test_x, test_y = [], []
    if options.test is None:
        idx = range(30) + range(170,200)
        test_x = data[idx]
        test_y = out[idx]
    else:
        test_x, test_y = readFile(options.test)


    ########################################################
    #Reshape
    if len(training_y.shape) == 1:
        training_y.shape = (len(training_y), 1)
        test_y.shape = (len(test_y), 1)

    if len(training_x.shape) == 1:
        training_x.shape = (len(training_x), 1)
        test_x.shape = (len(test_x), 1)

    #######################################################

    #Create mix of experts and set up hyper-params
    mixExperts = MixtureOfExperts(num_experts, training_x, training_y, test_x, test_y, poly_degree=1, feat_type="polynomial")

    #Train network and returns intermediate states for vizualisation
    mixExperts.trainNetwork(maxIterations, growing=False)
    mixExperts.setToBestParams()
    mixExperts.visualizePredictions(training_x, training_y, test_x, test_y)
