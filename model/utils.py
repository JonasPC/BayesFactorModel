# utilities functions
import pandas as pd


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
