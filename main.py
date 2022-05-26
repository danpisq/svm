from sklearn.model_selection import train_test_split

import pandas as pd
from sklearn.metrics import accuracy_score

from features import preprocess
from svm import SVM


if __name__ == '__main__':
    # load data
    data = pd.read_csv('data/data.csv')
    X, y = preprocess(data)

    X_train, X_valid, y_train, y_valid = train_test_split(X.to_numpy(), y.to_numpy(), test_size=0.2, random_state=42)

    svm = SVM(1000, 1e-5)
    svm.fit(X_train, y_train, max_epochs=10)
    tmp = y_valid
    y_predicted = svm.predict(X_valid)

    valid_accuracy = accuracy_score(y_valid, y_predicted)
    print(f'Validation accuracy {valid_accuracy}')
