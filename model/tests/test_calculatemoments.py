import pandas as pd
import numpy as np
import pytest

from model.calculatemoments import CalculateMoments


@pytest.fixture
def setup_mean_matrix1():
    x1 = [1 for i in range(1000)]
    x2 = [2 for i in range(1000)]
    x3 = [3 for i in range(1000)]
    return pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3})


@pytest.fixture
def setup_cm1(setup_mean_matrix1):
    """ calulate moments matrix 1"""
    return CalculateMoments(setup_mean_matrix1)


def test_calc_mean(setup_cm1):

    m1 = setup_cm1.mu()
    m2 = np.matrix([[1., 2., 3.]])

    np.testing.assert_array_equal(m1, m2)


def test_cov_shape(setup_cm1):

    assert setup_cm1.cov().shape == (3, 3)
