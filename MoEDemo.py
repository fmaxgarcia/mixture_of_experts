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

def function2D(x, y):
    out = list()
    for i in range(len(x)):
        zx = -2*(x[i]-5)**2 + 50
        zy = -2*(y[i]-5)**2 + 50

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


def normalize_data(test, val):
    dimensions = len(test.shape)
    if dimensions == 1:
        minx = min(test) if min(test) < min(val) else min(val)
        maxx = max(test) if max(test) > max(val) else max(val)
        test = (test - minx) / (maxx - minx)
        val = (val - minx) / (maxx - minx)
    else:
        for col in range(test.shape[1]):
            minx = min(test[:,col]) if min(test[:,col]) < min(val[:,col]) else min(val[:,col])
            maxx = max(test[:,col]) if max(test[:,col]) > max(val[:,col]) else max(val[:,col])
            test[:,col] = (test[:,col] - minx) / (maxx - minx) if maxx != minx else 1.0
            val[:,col] = (val[:,col] - minx) / (maxx - minx) if maxx != minx else 1.0
    return test, val




if __name__ == '__main__':
    #Options to run from command line
    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option("-r", "--rate", action="store", help="learning rate", type="float")
    parser.add_option("-g", "--gate", action="store", help="gate type [softmax - em]", type="choice", choices=['softmax', 'em'])
    parser.add_option("-n", "--num", action="store", help="number of experts", type="int")
    parser.add_option("-i", "--iter", action="store", help="maximum iterations", type="int")
    parser.add_option("-d", "--decay", action="store", help="decay rate", type="float")
    parser.add_option("-m", "--mode", action="store", help="mode [comp - competitive, coop - cooperative]", type="choice", choices=['comp', 'coop'])
    parser.add_option("-t", "--train", action="store", help="train set filename (none for synthetic data)")
    parser.add_option("-v", "--test", action="store", help="test set filename (none for synthetic data)")
    parser.add_option("-s", "--screen", action="store", help="screen [rec - record movie, screen - screenshot]", type="choice", choices=['rec', 'screen'])

    (options, args) = parser.parse_args()

    mode = "coop" if options.mode is None else options.mode
    gateType = "softmax" if options.gate is None else options.gate
    print "Running in mode: ", mode

    data, out = createData2D() if options.train is None else readFile(options.train)

    num_experts = 2 if options.num is None else options.num
    maxIterations = 100 if options.iter is None else options.iter
    learningRate = 0.01 if options.rate is None else options.rate
    decay = 0.98 if options.decay is None else options.decay
    rec = "screen" if options.screen is None else options.screen


    #Initialize and format training data
    training_y = out[30:170:1] if options.train is None else out
    training_x = data[30:170:1] if options.train is None else data

    testdata, actualOutput = [], []
    if options.test is None:
        idx = range(30) + range(170,200)
        testdata = data[idx]
        actualOutput = out[idx]
    else:
        testdata, actualOutput = readFile(options.test)

    ####Normalize test and validation data together###########
    ###########################################################

    training_x, testdata = normalize_data(training_x, testdata)
    training_y, actualOutput = normalize_data(training_y, actualOutput)

    ########################################################
    #Reshape
    if len(training_y.shape) == 1:
        training_y.shape = (len(training_y), 1)
        actualOutput.shape = (len(actualOutput), 1)

    if len(training_x.shape) == 1:
        training_x.shape = (len(training_x), 1)
        testdata.shape = (len(testdata), 1)

    #######################################################

    #Create mix of experts and set up hyper-params
    mixExperts = MixtureOfExperts(num_experts, gateType, mode, training_x, training_y, poly_degree=1, feat_type="polynomial")
    mixExperts.learningRate = learningRate
    mixExperts.decay = decay

    #Train network and returns intermediate states for vizualisation
    mixExperts.trainNetwork(training_x, training_y, testdata, actualOutput, maxIterations)

    mixExperts.setToBestParams()
    mixExperts.visualizePredictions(training_x, training_y, testdata, actualOutput, rec)
