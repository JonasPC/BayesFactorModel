import pandas as pd
import numpy as np


class CalculateMoments(object):

    """Calculates moments from a pd.DataFrame

    Parameters
    ==========
    df : pd.DataFrame
        dataframe with data from which moments should be calculated

    """

    def __init__(self, df):
        self.df = df
        self.data = np.matrix(df.T)

    def mu(self):
        ones = np.ones((1, self.data.shape[1]))
        return (1 / self.data.shape[1]) * (ones * self.data.T)

    def cov(self):
        return np.cov(self.data)
