import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
import math
import numpy as np
from pylab import *
from scipy.stats import multivariate_normal

class Plotter:

    def _create3Dmesh(self, x, y, mixtureOfExperts, output_index):
        xdata = sorted(x)
        ydata = sorted(y, reverse=True)
        X, Y = np.meshgrid(xdata, ydata)
        Z = np.zeros( X.shape )
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                xy = mixtureOfExperts.transform_features(np.asarray([X[i][j], Y[i][j]]))
                finalOutput, expertsOutputs = mixtureOfExperts.computeMixtureOutput( xy )
                if output_index > -1:
                    finalOutput = finalOutput[output_index]
                Z[i][j] = finalOutput

        return  X, Y, Z


    def plotPrediction(self, mixtureOfExperts, training_x, training_y, test_x, test_y):
        num_outputs = 1 if len(training_y.shape) == 1 else training_y.shape[1]

        for i in range(num_outputs):
            error, prediction = mixtureOfExperts.testMixture(test_x, test_y)
            figure = plt.figure()
            three_d = False if training_x.shape[1] == 1 else True
            plot1 = figure.add_subplot(111, projection='3d') if three_d else figure.add_subplot(111)
            cmap = colormaps()

            index = i if num_outputs > 1 else -1
            if three_d:
                xt, yt, zt = self._create3Dmesh(test_x[:,0], test_x[:,1], mixtureOfExperts, output_index=index)
                plot1.plot_surface(xt, yt, zt, cmap=cmap[7])
            else:
                plot1.scatter(training_x[:,0], training_y[:,i], c="red")

            string = "Error = %f" %(error) if type(error) is float else "Error = %f" %(error[i])
            ax = plot1.axis()
            x_range = ax[1] - ax[0]
            y_range = ax[3] - ax[2]
            if three_d:
                plot1.text(ax[0] + (x_range / 10), ax[3] - (y_range / 10), 0, string, fontsize = 12)
            else:
                plot1.text(ax[0] + (x_range / 10), ax[3] - (y_range / 10), string, fontsize = 12)

            plt.show()



    def plotExpertsPrediction(self, mixtureOfExperts, test_x, test_y):
        three_d = test_x.shape[1] != 1
        if three_d == False:
            num_outputs = 1 if len(test_y.shape) == 1 else test_y.shape[1]

            for i in range(num_outputs):
                predictions, y = [], []
                for e in mixtureOfExperts.experts:
                    predictions.append( list() )

                for x in test_x:
                    x = mixtureOfExperts.transform_features(x)
                    finalOutput, expertsOutputs = mixtureOfExperts.computeMixtureOutput(x)
                    y.append( finalOutput[i] )

                    for j in range(len(mixtureOfExperts.experts)):
                        predictions[j].append( expertsOutputs[j,i] )

                plt.subplot(121)
                for k in range(len(predictions)):
                    line, = plt.plot(test_x[:,0], predictions[k])
                    line.set_label("Expert %d" %(k))

                xs = test_x[:,0]
                line, = plt.plot(xs, y, 'ro')
                line.set_label("Overall Output")
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
                plt.show()



    def plotExpertsCenters(self, mixtureOfExperts, trainingdata, trainingoutput):
        figure = plt.figure()
        three_d = False if trainingdata.shape[1] == 1 else True
        means = np.zeros( (len(mixtureOfExperts.experts), mixtureOfExperts.experts[0].location.shape[0]) )

        if three_d:
            plot1 = figure.add_subplot(111)
            labels = list()
            for i in range(len(mixtureOfExperts.experts)):
                expertMean = mixtureOfExperts.experts[i].location
                means[i] = expertMean.T
                plot1.scatter(expertMean[1], expertMean[2], c="red")
                labels.append("Expert %d center" %(i))

            for label, x, y in zip(labels, means[:, 1], means[:, 2]):
                plt.annotate(
                label,
                xy = (x, y), xytext = (-20, 20),
                textcoords = 'offset points', ha = 'right', va = 'bottom',
                bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

        else:
            plot1 = figure.add_subplot(121)
            ax = plot1.axis()
            plot1.scatter(trainingdata[:,0], trainingoutput, c="red")
            ys = np.linspace(min(trainingoutput), max(trainingoutput), num=50)

            for i in range(len(mixtureOfExperts.experts)):
                expertMean = [ mixtureOfExperts.experts[i].location[1] ] * 50
                line, = plot1.plot(expertMean, ys)
                line.set_label("Expert %d center" %(i))

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
        plt.show()

    def plotGaussians(self, mixtureOfExperts, trainingdata, trainingoutput):
        three_d = False if trainingdata.shape[1] == 1 else True
        if three_d:
            figure = plt.figure()
            stepfirst = (max(trainingdata[:,0]) - min(trainingdata[:,0])) / 100.0
            stepsecond = (max(trainingdata[:,1]) - min(trainingdata[:,1])) / 100.0
            x, y = np.mgrid[min(trainingdata[:,0]):max(trainingdata[:,0]):stepfirst, min(trainingdata[:,1]):max(trainingdata[:,1]):stepsecond]
            pos = np.empty(x.shape + (2,))
            pos[:, :, 0] = x; pos[:, :, 1] = y
            ax2 = figure.add_subplot(111)
            for i, expert in enumerate(mixtureOfExperts.experts):
                mean = expert.location[1:].reshape(2,)
                rv = multivariate_normal(mean, mixtureOfExperts.gateNet.sigma[i,1:,1:])
                ax2.contourf(x, y, rv.pdf(pos),alpha=0.2)
        else:
            figure = plt.figure()
            plot1 = figure.add_subplot(121)
            min_out = min(trainingoutput)
            max_out = max(trainingoutput)
            trainingoutput = [(x - min_out) / (max_out- min_out) for x in trainingoutput]

            plot1.scatter(trainingdata[:,0], trainingoutput, c="red")
            ax = plot1.axis()
            xs = np.linspace(ax[0], ax[1], 100)
            for i in range(len(mixtureOfExperts.experts)):
                mean = mixtureOfExperts.experts[i].mean()[0,1]
                sigma = math.sqrt( mixtureOfExperts.gateNet.sigma[i][1][1] )
                line, = plot1.plot(xs, mixtureOfExperts.gateNet.alphas[i] * mlab.normpdf(xs, mean, sigma))
                line.set_label("Expert %d gaussian" %(i))

            plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
        plt.show()










