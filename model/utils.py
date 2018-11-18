# utilities functions
import pandas as pd
import numpy as np
from functools import wraps


def party_name_from_key(party_key):
    """returns the relevant party name"""
    relevant_parties = {0: 'Alternativet',
                        1: 'Dansk Folkeparti',
                        2: 'Det Konservative Folkeparti',
                        3: 'Enhedslisten - De Rød-Grønne',
                        4: 'Liberal Alliance',
                        5: 'Nye Borgerlige',
                        6: 'Radikale Venstre',
                        7: 'SF - Socialistisk Folkeparti',
                        8: 'Socialdemokratiet',
                        9: 'Venstre, Danmarks Liberale Parti'}

    return relevant_parties[party_key]


def print_dimension(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        for arg in args:
            try:
                print(arg.shape)
            except:
                pass
        return func(*args, **kwargs)
    return wrapper


@print_dimension
def matrix(x):
    return np.matrix(x)


def vector(array):
    return np.asarray(array).reshape(-1)


def read_clean_kv17(drop_party_key=False):
    """Reads the cleaned data from kv17

    Parameters
    ----------
    drop_party_key : (bool)
        If true the partyKey column is dropped from the returned DataFrame

    Returns
    -------
    type : (object) pd.DataFrame
        DataFrame with questions answered by politicians in the municipal elections 2017
    """

    df = pd.read_csv('data//clean_kv17.csv', index_col=0)

    if drop_party_key is True:
        df.drop('partyKey', axis=1, inplace=True)

    return df


def read_party_keys():
    df = pd.read_csv('data//clean_kv17.csv', index_col=0)
    return df['partyKey']


def read_testdata1():
    return pd.read_csv('data//test_data1.csv', index_col=0)


def read_testdata2():
    return pd.read_csv('data//test_data2.csv', index_col=0)
