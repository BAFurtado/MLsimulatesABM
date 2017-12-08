""" Experimenting with some models, def 'run_classifier' adapted from
http://scikit-learn.org/stable/auto_examples/ensemble/plot_voting_decision_regions.html
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize


def normalize_trial(x, xt):
    n = []
    for each in [x, xt]:
        n.append(normalize(each.as_matrix()))
    return n[0], n[1]


def run_classifiers(x, xt, y, yt):

    models = ['Tree', 'SVC', 'MPL', 'Voting']

    m1 = RandomForestClassifier(n_estimators=1000, criterion='gini', bootstrap=True, max_depth=15)
    m3 = SVC(C=1, kernel='poly', degree=3, probability=True)
    m4 = MLPClassifier(solver='lbfgs', early_stopping=True, activation='tanh', max_iter=200)
    voting = VotingClassifier(estimators=[('dt', m1), ('svc', m3), ('neural', m4)],
                              voting='soft')

    # Fitting models
    cls = [m1, m3, m4, voting]
    for each in cls:
        each.fit(x, y)

    models = dict(zip(models, cls))

    # Calculating accuracies and printing
    for key in models.keys():
        print('Score {}: {:.4f}.'.format(key, models[key].score(xt, yt)))
        # Examining confusion matrix
        yhat = models[key].predict(xt)
        cm = confusion_matrix(yt, yhat)
        print('Confusion Matrix {}:\n {}.'.format(key, cm))

    # Returns a dictionary of models' names and the model itself
    return models


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
    c, e, f, g = run_classifiers(X, XT, Y, YT)
