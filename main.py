from typing import Tuple
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import resample
import matplotlib.pyplot as plt

DATA_PATH = Path(__file__).resolve().parents[1] / "dataset" / "creditcard.csv"
data = pd.read_csv(DATA_PATH)

data = resample(data, n_samples=len(data[:100000]), replace=False, stratify=data, random_state=0)

print("DATA COLUMNS:", data.columns)
print("\nDATA SIZE:", data.shape)
print("\nFIRST 5 ROWS:", data.head)


def prepare_data(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_test = data
    X_train = data[data["Class"] != 1]

    y_test = X_test["Class"]
    y_train = X_train["Class"]

    X_test = X_test.drop(columns=["Class"])
    X_train = X_train.drop(columns=["Class"])

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_transformed_train = scaler.transform(X_train)
    X_transformed_test = scaler.transform(X_test)

    pca = PCA(n_components=30)
    pca.fit(X_transformed_train)
    X_transormed_train = pca.transform(X_transformed_train)
    X_transformed_test = pca.transform(X_transformed_test)

    return (X_transormed_train, y_train, X_transformed_test, y_test, scaler, pca)


def train_classical_model(X_train, y_train, X_test, y_test):
    clf_svm = OneClassSVM(kernel="rbf", degree=3, gamma=0.1, nu=0.01)
    clf_svm.fit(X_train)

    y_predict = clf_svm.predict(X_test)
    svm_predict = pd.Series(y_predict).replace([-1, 1], [1, 0])

    ps = precision_score(y_test, svm_predict)
    print("PRECISON", ps)
    rs = recall_score(y_test, svm_predict)
    print("RECALL", rs)

    cm = confusion_matrix(y_test, svm_predict)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()


(X_train, y_train, X_test, y_test, _, _) = prepare_data(data)

print("TRAIN", X_train[:5])
print("Y TRAIN", y_train[:5])

print("TEST", X_test[:5])
print("Y TEST", len(y_test[y_test[:] == 1]))

train_classical_model(X_train, y_train, X_test, y_test)
