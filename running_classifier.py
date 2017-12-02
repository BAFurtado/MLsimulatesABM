import os
import pickle

from numpy import set_printoptions
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle

import descriptive_stats
import generating_random_conf
import preparing_data

set_printoptions(precision=4)


def get_data(pathways, col1, col2):
    name = 'pre_processed_data\\' + pathways[-4:]
    if os.path.exists(name):
        with open(name, 'rb') as stored_data:
            p = pickle.load(stored_data)
            print('Loaded!')
        return p[0], p[1]
    else:
        x, y = preparing_data.main(pathways, col1, col2)
        with open(name, 'wb') as stored_data:
            pickle.dump([x, y], stored_data)
            print('Saved!')
        return x, y


def run_random_forest_cross(x, y):
    x, y = shuffle(x, y)
    m1 = RandomForestClassifier(n_estimators=100, criterion='entropy', bootstrap=True)
    accuracies = cross_val_score(m1, x, y['target'], cv=2)
    print('All accuracies RF Automatic')
    print(accuracies)
    print('Average accuracy Random Forest Automatic resampling {:.4f}'.format(sum(accuracies)/len(accuracies)))


def run_random_forest_split(x, y):
    m1 = RandomForestClassifier(n_estimators=100, criterion='entropy', bootstrap=True)
    train_size = int(len(x) * .67)
    m1.fit(x[train_size:], y['target'][train_size:])
    accuracy = m1.score(x[:train_size:], y['target'][:train_size])
    yhat = m1.predict(x[train_size:])
    cm = confusion_matrix(y[train_size:], yhat)
    print('Accuracy Random Forest Manual data-splitting {:.4f}'.format(accuracy))
    print('Confusion matrix for RF Manual')
    print(cm)
    print('Features importance: ')
    print('')
    out = dict()
    for i in range(len(m1.feature_importances_)):
        if m1.feature_importances_.item(i) > 0:
            out[x.columns[i]] = m1.feature_importances_.item(i)
    for w in sorted(out, key=out.get, reverse=True):
        print('{}: {:.4f}'.format(w, out[w]))
    return m1


def predict_random_forest_cross(a, b):
    return 1, 2


def main(x, y):
    descriptive_stats.print_conf_stats({'bases': [x], 'text':['actual']})
    run_random_forest_cross(x, y)
    model = run_random_forest_split(x, y)
    r = generating_random_conf.generate()
    s1, s0 = predict_random_forest_cross(r, model)
    descriptive_stats.print_conf_stats({'bases': [s1, s0], 'text': ['sim1', 'sim2']})


if __name__ == "__main__":
    cols_names = ['months', 'price_index', 'gdp_index', 'gdp_growth', 'unemployment', 'average_workers',
                  'families_wealth', 'families_savings', 'firms_wealth', 'firms_profit', 'gini_index',
                  'average_utility', 'inflation', 'average_qli']
    path = r'\\storage4\carga\MODELO DINAMICO DE SIMULACAO\Exits_python\JULY\SENSItivity\distributions'
    target1 = 'gdp_index'
    target2 = 'gini_index'
    a, b = get_data(path, target1, target2)
    main(a, b)
