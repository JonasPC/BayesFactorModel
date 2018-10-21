import numpy as np
from numpy import linalg
from scipy.stats import invgamma
import logging

from model.utils import vector


class GibbsSampler():

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    def __init__(self, n_factors, data):

        self.y = data
        self.k = n_factors
        self.p = data.shape[1]
        self.T = data.shape[0]
        self.I_k = np.identity(self.k)
        self.I_p = np.identity(self.p)

        self.v = 0.5
        self.s_sq = 2
        self.C0 = 2
        self.mu0 = 1

        self.Beta_list = list()
        self.Sigma_list = list()
        self.F_list = list()

        print('number of variables:', self.p, ' number of observations:', self.T)

    def f_t(self, Beta, Sigma, t):
        """Posterior of f_t"""

        y_t = self.y[[t]].T
        S_inv = linalg.inv(Sigma * self.I_p)

        scale = linalg.inv(self.I_k + np.dot(np.dot(Beta.T, S_inv), Beta))
        loc = vector(np.dot(np.dot(np.dot(scale, Beta.T), S_inv), y_t))

        return np.random.multivariate_normal(loc, scale)

    def d_i(self, Beta, F, i):
        """Helper method for calculating sigma"""

        y_i = self.y.T[[i]]
        Beta_i = Beta[[i]]

        tmp = (y_i.T - np.dot(F, Beta_i.T))
        return float(np.dot(tmp.T, tmp))

    def sigma_i(self, Beta, F, i):
        d_i = self.d_i(Beta, F, i)

        alpha = (self.v + self.T) / 2
        scale = (self.v * self.s_sq + d_i) / 2
        return invgamma.rvs(alpha, scale=scale)

    def Beta_i(self, Sigma, F, i):

        if i < self.k:

            C_i = self.C_i(F, Sigma, i)
            m_i = self.m_i(C_i, F, Sigma, i)

            B_i = np.random.multivariate_normal(m_i, C_i)
            while B_i[i] <= 0:
                # print(B_i[i])  # possible bug
                #B_i = np.random.multivariate_normal(m_i, C_i)
                B_i[i] = 0.1

            if i < self.k:
                B_i = np.append(B_i, np.zeros(self.k - i - 1))

        elif i >= self.k:

            C_k = self.C_k(F, Sigma, i)
            m_k = self.m_k(C_k, F, Sigma, i)

            B_i = np.random.multivariate_normal(m_k, C_k)

        else:
            raise ValueError('k is {0}, i is {1} - Beta_i probs'.format(self.k, i))

        return vector(B_i)

    def C_i(self, F, Sigma, i):
        """If i <= k """

        F_i = F.T[:i + 1].T
        sigma_i = Sigma[i]
        identity_i = np.identity(i + 1)

        return linalg.inv((1 / self.C0) * identity_i + (1 / sigma_i) * np.dot(F_i.T, F_i))

    def C_k(self, F, Sigma, i):
        """if i > k"""

        sigma_i = Sigma[i]
        return linalg.inv((1 / self.C0) * self.I_k + (1 / sigma_i) * np.dot(F.T, F))

    def m_i(self, C_i, F, Sigma, i):
        """If i <= k """

        F_i = F[:, :i + 1]  # 2000 X i
        sigma_i = Sigma[i]  # 1 x 1
        ones_i = np.matrix(np.ones(i + 1)).T
        y_i = self.y[:, [i]]
        tmp = (1 / self.C0) * self.mu0 * ones_i + (1 / sigma_i) * np.dot(F_i.T, y_i)
        return vector(np.dot(C_i, tmp))

    def m_k(self, C_k, F, Sigma, i):
        """if i > k"""

        sigma_i = Sigma[i]  # 1 x 1
        ones_k = np.matrix(np.ones(self.k)).T
        y_i = self.y[:, [i]]

        tmp = (1 / self.C0) * self.mu0 * ones_k + (1 / sigma_i) * np.dot(F.T, y_i)
        return vector(np.dot(C_k, tmp))

    def calc_Beta(self):

        B = np.matrix([self.Beta_i(self.Sigma, self.F, i) for i in range(self.p)])
        self.Beta_list.append(B)
        return B

    def calc_F(self):

        F = np.matrix([self.f_t(self.Beta, self.Sigma, t) for t in range(self.T)])
        self.add('F', F)
        return F

    def calc_Sigma(self):

        Sigma = vector([self.sigma_i(self.Beta, self.F, i) for i in range(self.p)])
        self.add('Sigma', Sigma)
        return Sigma

    @property
    def Beta(self):
        return self.Beta_list[-1]

    @property
    def F(self):
        return self.F_list[-1]

    @property
    def Sigma(self):
        return self.Sigma_list[-1]

    def add(self, param, value):
        """ add to Sigma_list, Beta_list or F_list

        Parameters
        ==========
        param: (str)
            string that should be of {'Sigma', 'F', 'Beta'}
        value: (obj)
            appropriate object for given list

        """

        if param == 'Sigma':
            self.Sigma_list.append(value)

        elif param == 'F':
            self.F_list.append(value)

        elif param == 'Beta':
            self.Beta_list.append(value)

        else:
            raise ValueError("Param must be in {'F', 'Sigma', 'Beta'}")

    def sampler(self, n_iterations):

        logging.info("Sampling begins")
        for i in range(n_iterations):

            self.calc_F()
            self.calc_Sigma()
            self.calc_Beta()
            if (i % 10 == 0):
                logging.info("run {0} simulations".format(i))
