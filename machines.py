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
def run_classifiers(x, y):
    m1 = DecisionTreeClassifier(max_depth=7)
    m2 = KNeighborsClassifier(n_neighbors=3)
    m3 = SVC(C=1, kernel='poly', degree=3, probability=True)
    m4 = MLPClassifier(solver='lbfgs', early_stopping=True, activation='tanh')
    voting = VotingClassifier(estimators=[('dt', m1), ('knn', m2), ('svc', m3), ('neural', m4)], voting='soft')
    m1.fit(x, y)
    m2.fit(x, y)
    m3.fit(x, y)
    m4.fit(x, y)
    voting.fit(x, y)
    for each in [m1, m2, m3, m4, voting]:
        print('Score: {:.4f}'.format(each.score(x, y)))
    return m1, m2, m3, m4, voting


def run_random_forest_split(x, y):
    x, y = shuffle(x, y)
    m1 = RandomForestClassifier(n_estimators=1000, criterion='entropy', bootstrap=True)
    train_size = int(len(x) * .7)
    m1.fit(x[train_size:], y['target'][train_size:])
    accuracy = m1.score(x[:train_size:], y['target'][:train_size])
    print('Accuracy Random Forest Manual data-splitting {:.4f}'.format(accuracy))
    return m1


def predict(model, data):
    return model.predict(data)


if __name__ == "__main__":
    import main
    path = r'\\storage4\carga\MODELO DINAMICO DE SIMULACAO\Exits_python\JULY'
    target1 = 'gdp_index'
    target2 = 'gini_index'
    a, b = main.get_data(path, target1, target2)
    c, d, e, f, g = run_classifiers(a, b)
