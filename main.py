import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def read_data():
    dataset = pd.read_csv('data.csv')
    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())
    dataset['EstimatedSalary'] = dataset['EstimatedSalary'].fillna(dataset['EstimatedSalary'].mean())
    X = dataset.iloc[:len(dataset), [2, 3]].values
    y = dataset.iloc[:len(dataset), -1].values
    return X, y


def sigmoide(z):
    return 1. / (1. + np.exp(-np.clip(z, -250, 250)))


def net_input(X, theta):
    return np.dot(X, theta[1:]) + theta[0]


def logistic_regression(X, y, alpha=0.1, iterations=1000, random_state=1):
    rgen = np.random.RandomState(random_state)
    theta = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
    for i in range(iterations):
        z = net_input(X, theta)
        h = sigmoide(z)
        error = y - h
        gradient = np.dot(X.T, error)
        theta[1:] += - alpha * gradient
        theta[0] += - alpha * np.sum(error)
    return theta


def predict(X, theta):
    return np.where(sigmoide(net_input(X, theta)) >= 0.5, 0, 1)


def plot_confusion_matrix(y_test, y_pred):
    data = {'y_Actual': y_test, 'y_Predicted': y_pred}
    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
    sn.heatmap(confusion_matrix, annot=True)
    plt.show()


def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.plot(X[y == 0, 0], X[y == 0, 1], 'ro', label='0')
    plt.plot(X[y == 1, 0], X[y == 1, 1], 'bo', label='1')
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                            np.arange(x2_min, x2_max, resolution))
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.4)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend(loc='upper left')
    plt.show()


class FuncionWrapper:
    def __init__(self, func, theta):
        self.func = func
        self.theta = theta

    def predict(self, X):
        return self.func(X, self.theta)


def main():
    x, y = read_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
    sc_X = StandardScaler()
    x_train = sc_X.fit_transform(x_train)
    x_test = sc_X.transform(x_test)
    theta = logistic_regression(x_train, y_train, alpha=0.0000001, iterations=200000, random_state=0)
    y_pred = predict(x_test, theta)
    plot_confusion_matrix(y_test, y_pred)
    plot_decision_regions(x_test, y_test, classifier=FuncionWrapper(predict, theta))

    logistic = LogisticRegression(random_state=0)
    logistic.fit(x_train, y_train)
    y_predict_py = logistic.predict(x_test)
    plot_confusion_matrix(y_test, y_predict_py)
    plot_decision_regions(x_test, y_test, classifier=logistic)


if __name__ == '__main__':
    main()

