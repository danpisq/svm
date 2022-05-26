from typing import Tuple

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def preprocess(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # drop not needed columns
    data.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
    # map target column to 1 and -1 labels
    data['diagnosis'] = data['diagnosis'].map({'M': 1.0, 'B': -1.0})

    y = data['diagnosis']
    X = data.drop(columns=['diagnosis'])

    # normalise data
    X_normalized = MinMaxScaler().fit_transform(X.values)
    X = pd.DataFrame(X_normalized)
    return X, y