""" Experimenting with some models
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle


# Training classifiers (see note above)
def run_classifiers(x, xt, y, yt):

    models = ['Tree', 'KNeighbors', 'SVC', 'MPL', 'Voting']

    m1 = DecisionTreeClassifier(max_depth=7)
    m2 = KNeighborsClassifier(n_neighbors=3)
    m3 = SVC(C=1, kernel='poly', degree=3, probability=True)
    m4 = MLPClassifier(solver='lbfgs', early_stopping=True, activation='tanh')
    voting = VotingClassifier(estimators=[('dt', m1), ('knn', m2), ('svc', m3), ('neural', m4)],
                              voting='soft')

    # Fitting models
    cls = [m1, m2, m3, m4, voting]
    for each in cls:
        each.fit(x, y)

    models = dict(zip(models, cls))

    # Calculating accuracies and printing
    for key in models.keys():
        print('Score {}: {:.4f}.'.format(key, models[key].score(xt, yt)))

    # Returns a dictionary of models' names and the model itself
    return models


def run_random_forest_split(x, xt, y, yt):
    x, y = shuffle(x, y)
    m1 = RandomForestClassifier(n_estimators=1000, criterion='entropy', bootstrap=True)
    m1.fit(x, y)
    accuracy = m1.score(xt, yt)
    print('Accuracy Random Forest {:.4f}'.format(accuracy))
    return m1


def predict(model, data):
    return model.predict(data)


def fit(model, a, b):
    model.fit(a, b)


if __name__ == "__main__":
    import main
    path = r'\\storage4\carga\MODELO DINAMICO DE SIMULACAO\Exits_python\JULY'
    target1 = 'gdp_index'
    target2 = 'gini_index'
    X, XT, Y, YT = main.get_data(path, target1, target2)
    c, d, e, f, g = run_classifiers(X, XT, Y, YT)
