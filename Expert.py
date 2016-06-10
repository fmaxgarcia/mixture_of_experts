import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from FourierFeatures import FourierFeatures
import math

class Expert:

    def _transform_poly_features(self, feat):
        if len(feat.shape) == 1:
            feat = feat.reshape( (1, feat.shape[0]) )
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


    def __init__(self, training_x, training_y, location, expertIndex, poly_degree=1, feat_type="polynomial"):
        self.poly_degree = poly_degree
        self.feat_type = feat_type
        _training_x = self.transform_features(training_x)
        self.weights = np.random.random( (training_y.shape[1], _training_x.shape[1]) )
        self.sigma = np.identity(_training_x.shape[1]) / 100.0
        self.dim_input = _training_x.shape[1]
        self.dim_output = training_y.shape[1]

        diff = math.fabs(_training_x.shape[1] - location.shape[0])
        if location.shape[0] < _training_x.shape[1]:
            location = np.vstack( (location, np.zeros((int(diff), 1))) )
        elif location.shape[0] > _training_x.shape[1]:
            for i in range(int(diff)):
                location = np.delete(location, location.shape[1]-1, 0)

        self.location = location
        for i in range(training_y.shape[1]):
            for j in range(_training_x.shape[1]):
                self.weights[i][j] = np.random.uniform(-1,1)

        self.error = 0.0
        self.sum_hs = 0.0
        self.best_weights = self.weights.copy()
        self.best_location = self.location.copy()
        self.best_sigma = self.sigma.copy()
        self.index = expertIndex

    def mean(self):
        if len(self.location.shape) == 1:
            self.location = self.location.reshape( (self.location.shape[0], 1) )
        return np.transpose(self.location)

    def setWeights(self, weights):
        diff = math.fabs(weights.shape[1] - self.weights.shape[1])
        if weights.shape[1] > self.weights.shape[1]:
            for i in range(int(diff)):
                weights = np.delete(weights, weights.shape[1]-1, 1)
        elif weights.shape[1] < self.weights.shape[1]:
            weights = np.hstack( (weights, np.random.random((self.dim_output, int(diff)))) )


        self.weights = weights


    def setMean(self, mean):
        if len(mean.shape) == 1:
            mean = mean.reshape( (1, mean.shape[0]))
        loc = np.transpose(mean)
        diff = math.fabs(loc.shape[0] - self.location.shape[0])
        if loc.shape[0] > self.location.shape[0]:
            for i in range(int(diff)):
                loc = np.delete(loc, loc.shape[1]-1, 0)
        elif loc.shape[0] < self.location.shape[0]:
            loc = np.vstack( (loc, np.random.random((int(diff), 1))) )


        self.location = loc

    def resetError(self):
        self.error = 0.0
        self.sum_hs = 0.0

    def addError(self, hs, error_sqr):
        self.error += hs * error_sqr
        self.sum_hs += hs

    def normalizeError(self):
        if self.sum_hs == 0:
            self.sum_hs = 0.00001
        self.error = self.error / self.sum_hs
        # else:
        #     print "Expert sum_hs is 0. Cannot normalize error"

    def computeExpertyhat(self, x):
        x = self.transform_features(x)
        result = self.weights.dot( x.T )
        if len(result.shape) == 2: #when last dimension is 1
            result = result.reshape( (result.shape[0],) )
        if math.isnan(result[0]):
            print "Exception: X ", x, "\nWeights ", self.weights, "\nResult ", result
            raise
        return result

    def saveBestWeights(self):
        self.best_weights = self.weights.copy()
        self.best_location = self.location.copy()
        self.best_sigma = self.sigma.copy()

    def setToBestWeights(self):
        self.weights = self.best_weights.copy()
        self.location = self.best_location.copy()
        self.sigma = self.best_sigma.copy()

