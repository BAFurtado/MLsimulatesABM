""" def 'run_classifer' code adapted from
http://scikit-learn.org/stable/auto_examples/ensemble/plot_voting_decision_regions.html
# sphx-glr-auto-examples-ensemble-plot-voting-decision-regions-py
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
    m1.fit(x, y)
    m2.fit(x, y)
    m3.fit(x, y)
    m4.fit(x, y)
    voting.fit(x, y)

    cls = [m1, m2, m3, m4, voting]
    for i in range(len(cls)):
        print('Score {}: {:.4f}'.format(models[i], cls[i].score(xt, yt)))
    return m1, m2, m3, m4, voting


def run_random_forest_split(x, xt, y, yt):
    x, y = shuffle(x, y)
    m1 = RandomForestClassifier(n_estimators=1000, criterion='entropy', bootstrap=True)
    m1.fit(x, y)
    accuracy = m1.score(xt, yt)
    print('Accuracy Random Forest {:.4f}'.format(accuracy))
    return m1


def predict(model, data):
    return model.predict(data)


if __name__ == "__main__":
    import main
    path = r'\\storage4\carga\MODELO DINAMICO DE SIMULACAO\Exits_python\JULY'
    target1 = 'gdp_index'
    target2 = 'gini_index'
    x, xt, y, yt = main.get_data(path, target1, target2)
    c, d, e, f, g = run_classifiers(x, xt, y, yt)
