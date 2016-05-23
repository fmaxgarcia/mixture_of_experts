import numpy as np
import math
from GaussianGate import GaussianGate
from Expert import Expert
from Plotter import Plotter

from sklearn.preprocessing import PolynomialFeatures
from FourierFeatures import FourierFeatures

def normalize_data(test, val):
    dimensions = len(test.shape)
    mins, maxs = [], []
    if dimensions == 1:
        minx = min(test) if min(test) < min(val) else min(val)
        maxx = max(test) if max(test) > max(val) else max(val)
        test = (test - minx) / (maxx - minx)
        val = (val - minx) / (maxx - minx)
        mins.append( minx )
        maxs.append( maxx )
    else:
        for col in range(test.shape[1]):
            minx = min(test[:,col]) if min(test[:,col]) < min(val[:,col]) else min(val[:,col])
            maxx = max(test[:,col]) if max(test[:,col]) > max(val[:,col]) else max(val[:,col])
            test[:,col] = (test[:,col] - minx) / (maxx - minx) if maxx != minx else 1.0
            val[:,col] = (val[:,col] - minx) / (maxx - minx) if maxx != minx else 1.0
            mins.append( minx )
            maxs.append( maxx )
    return test, val, np.asarray(mins), np.asarray(maxs)


class MixtureOfExperts:

    def _transform_poly_features(self, feat):
        if self.poly_degree > 1:
            polynomial = PolynomialFeatures(self.poly_degree)
            feat = polynomial.fit_transform(feat)
        else:
            if len(feat.shape) > 1:
                ones = np.ones( (feat.shape[0], 1) )
                feat = np.append(ones, feat, axis=1)
            else:
                ones = np.ones( (1,) )
                feat = np.append(ones, feat, axis=0)
        return feat

    def _transform_fourier_features(self, feat):
        return FourierFeatures(self.poly_degree, feat)

    def transform_features(self, feat):
        if self.feat_type == "polynomial":
            return  self._transform_poly_features(feat)
        elif self.feat_type == "fourier":
            return self._transform_fourier_features(feat)
        else:
            return feat


    def __init__(self, num_experts, training_x, training_y, test_x, test_y, poly_degree=1, feat_type="polynomial"):
        self.poly_degree = poly_degree
        self.feat_type = feat_type

        self.norm_training_x, self.norm_test_x, self.x_mins, self.x_maxs = normalize_data(training_x, test_x)
        self.norm_training_y, self.norm_test_y, self.y_mins, self.y_maxs = normalize_data(training_y, test_y)

        training_x = self.transform_features(training_x)
        dimension_in = training_x.shape[1]
        dimension_out = training_y.shape[1]
        self.gateNet = GaussianGate(num_experts, dimension_in, dimension_out)
        self.training_iterations = 0
        self.experts = list()
        self.numExperts = num_experts
        self.bestError = float("inf")
        for i in range(num_experts):
            location = np.ones( (dimension_in, 1) )
            for j in range(dimension_in):
                if j == 0: continue
                maxx = max(training_x[:,j])
                minx = min(training_x[:,j])
                step = (maxx - minx) / num_experts
                location[j] = (minx + step/2) + step * i

            self.experts.append( Expert(dimension_in, dimension_out, location, i) )

    def computeExpertsOutputs(self, x):
        return np.array([e.computeExpertyhat(x) for e in self.experts]).T
        

    def computeMixtureOutput(self, x):
        expertOutputs = self.computeExpertsOutputs(x)
        gateOutputs = self.gateNet.outputs(self.experts, x)
        finalOutput = expertOutputs.dot( gateOutputs )

        return finalOutput, expertOutputs

    def testMixture(self, test_x, test_y, recordErrors=False):

        finalOutputs = list()
        transformed_test_x = self.transform_features(test_x)
        for i, x in enumerate(transformed_test_x):
            mixPrediction, expPrediction = self.computeMixtureOutput(x)
            finalOutputs.append( mixPrediction )

            ########### Grow netowork ############
            ##### Ramamurti equation 8 ###############
            if isinstance(self.gateNet, GaussianGate) and recordErrors:
                y = test_y[i]
                hs = self.gateNet._compute_hs(self.experts, x, y)
                for j, expert in enumerate(self.experts):
                    expert.addError( hs[j], sum((y - expPrediction[:,j])**2) / test_y.shape[0] )

        error = 0
        for i in range(len(finalOutputs)):
            error += ((test_y[i] - finalOutputs[i])**2) / transformed_test_x.shape[0]

        return error, finalOutputs 


    def growNetwork(self, sortedExperts):
        transformed_training_x = self.transform_features(self.norm_training_x)
        for expert in sortedExperts:
            meanBackup = expert.mean().copy()
            alphaBackup = self.gateNet.alphas[expert.index].copy()
            sigmaBackup = self.gateNet.sigma[expert.index].copy()
            weightsBackup = expert.weights.copy()

            newExpert = Expert(self.norm_training_x.shape[1], self.norm_training_y.shape[1], expert.location, self.numExperts)
            newExpert.weights = expert.weights.copy()

            ###########Update experts according to Ramamurti page 5######################
            #############################################################################
            firstMean, secondMean = self.gateNet.find_best_means(expert, transformed_training_x, self.norm_training_y)
            expert.setMean(firstMean)
            newExpert.setMean(secondMean)

            newAlpha =  np.array([alphaBackup / 2.0])
            self.gateNet.alphas = np.vstack( (self.gateNet.alphas, newAlpha) )
            self.gateNet.alphas[expert.index] /= 2.0

            newSigma = np.array( [sigmaBackup] )
            self.gateNet.sigma = np.vstack( (self.gateNet.sigma, newSigma) )


            newMean, oldMean = self.gateNet.weighted_2_means(transformed_training_x, self.norm_training_y, newExpert, expert)
            newExpert.setMean(newMean)
            expert.setMean(oldMean)


            self.experts.append(newExpert)
            ##############################################################################
            #################### Test new network ########################################

            for i in range(5):
                self.gateNet.train( transformed_training_x, self.norm_training_y, self.experts)
                error, prediction = self.testMixture(self.norm_test_x, self.norm_test_y)
                avg_error = sum(error) / len(error)
                if  self.bestError - avg_error > 0.0001:
                    print "Error ", avg_error
                    print "Errors: ", error
                    self._saveParameters(avg_error)
                    self.numExperts += 1
                    print "Adding new expert!"
                    return True

            ###### Revert to previous network########
            expert.setMean(meanBackup)
            expert.weights = weightsBackup
            self.gateNet.alphas[expert.index] = alphaBackup
            self.gateNet.sigma[expert.index] = sigmaBackup
            self.experts.remove(newExpert)
            self.gateNet.sigma = np.delete(self.gateNet.sigma, self.numExperts, 0)
            self.gateNet.alphas = np.delete(self.gateNet.alphas, self.numExperts, 0)
            print "Reversing to previous network"

        return False

    def _saveParameters(self, error):
        #
        # print "Saving best parameters"
        self.bestError = error
        self.gateNet.saveBestParams()
        [e.saveBestWeights() for e in self.experts]


    def trainNetwork(self, maxIterations, growing=False):
        errors = list()
        keepGrowing = True
        transformed_training_x = self.transform_features(self.norm_training_x)

        while keepGrowing:
            iterations = 0
            while True:

                self.gateNet.train( transformed_training_x, self.norm_training_y, self.experts)

                [e.resetError() for e in self.experts]
                last_error, prediction = self.testMixture(self.norm_test_x, self.norm_test_y, recordErrors=True)
                avg_error = sum(last_error) / len(last_error)

                print "Errors: ", last_error
                print "Error: ", avg_error
                if avg_error < self.bestError:
                    self._saveParameters(avg_error)

                errors.append(avg_error)

                error_change = 1 if self.training_iterations < 5 else math.fabs(errors[self.training_iterations-1] - errors[self.training_iterations])
               # print "CHANGE ", error_change
                self.training_iterations += 1
                iterations += 1
                if iterations == maxIterations:
                   print "Max iterations reached"
                   break
                if error_change < .00001:
                    print "Min Error"
                    break

            [e.normalizeError() for e in self.experts]
            errorSorted = sorted(self.experts, key=lambda x:x.error, reverse=True)

            if self.bestError < 0.0001:
                break
            keepGrowing = False if growing == False else self.growNetwork(errorSorted)
        print "Final network %d experts" %(len(self.experts))


    def setToBestParams(self):
        self.gateNet.setToBestParams()
        for e in self.experts:
            e.setToBestWeights()



    def visualizePredictions(self, training_x, training_y, test_x, test_y):
        #Input 1 or 2 D. Output 1 D.
        assert (training_x.shape[1] == 1 or training_x.shape[1] == 2) and training_y.shape[1] == 1, \
            "Invalid Dimensions for Plotting Results - Input: %d - Output: %d" %(training_x.shape[1], training_y.shape[1])

        plotter = Plotter()
        plotter.plotPrediction(self, training_x, training_y, self.norm_test_x, self.norm_test_y)
        plotter.plotExpertsPrediction(self, test_x, test_y)
        plotter.plotExpertsCenters(self, training_x, training_y)
        plotter.plotGaussians(self, training_x, training_y)









