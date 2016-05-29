import numpy as np
import math
from scipy import optimize


class GaussianGate():
    def __init__(self, num_experts, dimensions_in, dimensions_out):
        self.alphas = np.random.random( (num_experts, 1) )
        self.sigmas = []

        #For visualization only
        self.tracking_alphas = list()
        self.tracking_sigmas = list()
        self.tracking_ms = list()
        self.experts_weights = list()


        self.saveBestParams()

    def saveBestParams(self):
        self.best_alphas = self.alphas.copy()

    def setToBestParams(self):
        self.alphas = self.best_alphas.copy()

    def outputs(self, experts, x):
        n = x.shape[0]
        px_times_alpha = list()
        for expert in experts:
            sigma_i = expert.sigma
            m_i = expert.mean()
            transformed_x = expert.transform_features(x)
            
            xm_diff = transformed_x - m_i
            mult = -0.5 * (xm_diff.dot(np.linalg.pinv(sigma_i)).dot(np.transpose(xm_diff)))
            prod = math.pow((2*math.pi), -n/2.0) * np.power(np.linalg.det(sigma_i), -1.0/2.0)
            if mult < -200.0:
                mult = -200.0
            px_given_vj = prod * np.exp(mult)
            px_times_alpha.append( (px_given_vj * self.alphas[expert.index]) )


        params = np.random.random( (len(experts), 1) )
        sum_px = sum(px_times_alpha)
        for i in range(params.shape[0]):
            params[i] = px_times_alpha[i] / sum_px

        return np.transpose(params)[0]

    #E-step
    def _compute_hs(self, experts, x, y):
        if len(x.shape) > 1:
            x = x[0]
            y = y[0]

        expertOutput = np.array([e.computeExpertyhat(x) for e in experts])
        g_xv = self.outputs(experts, x)
        y = np.transpose(y)
        if expertOutput[0].shape != y.shape:
            raise Exception("Output shape does not align. Output shape: ", expertOutput.shape, " Y: ", y.shape)

        exponents = np.array([-0.5 * np.transpose((y - expertOutput[i])).dot((y - expertOutput[i])) for i in xrange(len(experts))])
        exponents[ exponents < -200.0 ] = -200.0
        temp_hs = (g_xv * np.exp( exponents )).reshape( (len(experts), 1) )

        return temp_hs / np.sum(temp_hs, axis=0)

    #M-step
    def _update_alphas(self, experts, xs, ys):
        n = xs.shape[0] 
        sum_hs = np.zeros( (len(experts), 1) )
        for i in xrange(ys.shape[0]):
            sum_hs += self._compute_hs(experts, xs[i], ys[i])

        return sum_hs / n

    def _update_ms(self, experts, xs, ys):
        sum_hs = np.zeros( (len(experts), 1) )
        sum_hs_x = []
        for i in xrange(ys.shape[0]):
            hs = self._compute_hs(experts, xs[i], ys[i])
            sum_hs += hs
            for j in range(len(experts)):
                transformed_x = experts[j].transform_features(xs[i])
                if len(sum_hs_x) < len(experts):
                    sum_hs_x.append( hs[j] * transformed_x )
                else:
                    sum_hs_x[j] += hs[j] * transformed_x

        ms = list()
        for i in xrange(len(experts)):
            ms.append( sum_hs_x[i] / sum_hs[i] )

        return ms


    def _update_sigma(self, experts, xs, ys):
        xshape = (1, xs[0].shape[0])
        yshape = (1, ys[0].shape[0])
        sum_hs = np.zeros( (len(experts), 1) )
        sum_hs_xm = []
        for expert in experts:
            sum_hs_xm.append( np.zeros(expert.sigma.shape) )

        for i in range(ys.shape[0]):
            x, y = xs[i], ys[i]
            x.shape = xshape
            y.shape = yshape
            hs = self._compute_hs(experts, x, y)
            sum_hs += hs
            for j in range(hs.shape[0]):
                m = experts[j].mean()
                x_transformed = experts[j].transform_features(x)
                hj = hs[j]
                xm_xmt = np.transpose((x_transformed - m)).dot((x_transformed - m))
                sum_hs_xm[j] += hj * xm_xmt

        sigmas = list()
        for i in range(len(experts)):
            sigmas.append( sum_hs_xm[i] / sum_hs[i]  )

        return sigmas


    def find_best_means(self, expert, training_x, training_y):
        vals = list()
        for i in range(training_x.shape[0]):
            x = training_x[i]
            y = training_y[i]
            hs = self._compute_hs([ expert ], x, y)
            transformed_x = expert.transform_features(x)
            vals.append( (hs[0], transformed_x) )

        vals = sorted(vals, key=lambda x:x[0][0], reverse=True)
        return vals[0][1], vals[1][1]


    def weighted_2_means(self, training_x, training_y, newExpert, oldExpert):
        htsum = np.zeros( (2, 1) )
        htxsum = [ np.zeros( (1, newExpert.dim_input)), np.zeros((1, oldExpert.dim_input)) ]

        for i in range(training_x.shape[0]):
            x = training_x[i]
            y = training_y[i]

            hs = self._compute_hs( [ newExpert, oldExpert ], x, y)


            if hs[0] > hs[1]:
                htsum[0] += hs[0]
                htxsum[0] += hs[0] * newExpert.transform_features(x)
            else:
                htsum[1] += hs[1]
                htxsum[1] += hs[1] * oldExpert.transform_features(x)

        if htsum[0] == 0.0:
            htsum[0] += 0.1; htsum[1] -= 0.1
        elif htsum[1] == 0.0:
            htsum[1] += 0.1; htsum[0] -= 0.1

        return htxsum[0] / htsum[0], htxsum[1] / htsum[1]


    def update_weights_wls(self, experts, training_x, training_y):
        Cmat = np.zeros( (len(experts), training_x.shape[0], training_x.shape[0]) )
        yvec = np.zeros( (training_y.shape[0], 1) )
        all_weights = []

        for j, expert in enumerate(experts):
            Amat = np.zeros( (training_x.shape[0], expert.dim_input) )
            return_weights = []
            for k in range(training_y.shape[1]):
                for i in range(training_x.shape[0]):
                    x = training_x[i]
                    y = training_y[i]

                    transformed_x = expert.transform_features(x)
                    Amat[i] = transformed_x

                    yvec[i] = y[k]
                    hs = self._compute_hs(experts, x, y)
                    for l in range(len(experts)):
                        Cmat[l][i][i] = hs[l]
                    # if hs[j] == 0:
                    #     print "Weights cannot be 0!!!!"
                try:
                    weights = np.linalg.pinv( np.transpose(Amat).dot(Cmat[j]).dot(Amat) ).dot(np.transpose(Amat)).dot(Cmat[j]).dot(yvec)
                    return_weights.append( weights )
                except np.linalg.LinAlgError:
                    print "Linalg Error"
            all_weights.append( return_weights )

        return all_weights


    def train(self, training_x, training_y, experts):

        new_alphas = self._update_alphas(experts, training_x, training_y)
        new_ms = self._update_ms(experts, training_x, training_y)
        new_sigmas = self._update_sigma(experts, training_x, training_y)

        new_weights = self.update_weights_wls(experts, training_x, training_y)


        for i, expert in enumerate(experts):
            weights = np.asarray(new_weights[i])
            if len(weights.shape) == 3:
                weights = np.squeeze(weights, axis=2)
            expert.weights = weights

        self.alphas = new_alphas

        for i, expert in enumerate(experts):
            new_sigma_i = new_sigmas[i] * np.eye( new_sigmas[i].shape[0] )
            new_sigma_i[ new_sigma_i < 0.00001 ] = 0.00001
            expert.sigma = new_sigma_i

        for i, m in enumerate(new_ms):
            experts[i].setMean(m)

        thetas = []
        for i, expert in enumerate(experts):
            thetas.append( expert.weights )


        #for visualization only
        self.tracking_alphas.append( self.alphas.copy() )
        self.tracking_sigmas.append( [e.sigma.copy() for e in experts] )
        self.tracking_ms.append( [e.mean().copy() for e in experts] )
        self.experts_weights.append( thetas )


    def setIterationValues(self, iteration, experts):
        self.alphas = self.tracking_alphas[iteration]

        for i, expert in enumerate(experts):
            expert.weights = self.experts_weights[iteration][i]
            expert.sigma = self.tracking_sigmas[iteration][i]
            expert.setMean( self.tracking_ms[iteration][i] )
