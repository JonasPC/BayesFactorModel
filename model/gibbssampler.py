
class GibbsSampler(self, data, number_of_factors=3):

    """
    Samples parameters of posterior given data and hyper parameters.

    ===
    - Parameters -
    data: Must be a df of dimension (N x p)
    number_of_factors: must be a series
    """
    import numpy as np

    def __init__(self):
        self.data = data
        self.n = len(data) #num of observations
        self.k = len(self.data.columns) #num of covariates
        self.p = len(number_of_factors) #num of factors

    def cov_matrix(self, type='identity'):
        """
        Sets the cov_matrix to a fixed identity matrix or allows it to
        be simulated.
        """
        if type == 'identity':
            self.cov = np.identity(self.n)
        elif type == 'simulate':
            self.cov = pd.DataFrame(np.ones((self.n, self.n)))
        else:
            raise Exception("type must be 'identity' or 'simulate'")
