import os
import pickle

from numpy import set_printoptions
from sklearn.ensemble import RandomForestClassifier
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
    return m1


def run_random_forest_split(x, y):
    m1 = RandomForestClassifier(n_estimators=100, criterion='entropy', bootstrap=True)
    train_size = int(len(x) * .67)
    m1.fit(x[train_size:], y['target'][train_size:])
    accuracy = m1.score(x[:train_size:], y['target'][:train_size])
    print('Accuracy Random Forest Manual data-splitting {:.4f}'.format(accuracy))
    return m1


def predict_random_forest_cross(model, data):
    return model.predict(data)


def main(x, y):
    descriptive_stats.print_conf_stats({'bases': [x], 'text': ['actual']})
    m2 = run_random_forest_split(x, y)
    r = generating_random_conf.compound()
    descriptive_stats.print_conf_stats({'bases': [r], 'text':['generated']})
    y2 = predict_random_forest_cross(m2, r[x.columns.tolist()])


if __name__ == "__main__":
    cols_names = ['months', 'price_index', 'gdp_index', 'gdp_growth', 'unemployment', 'average_workers',
                  'families_wealth', 'families_savings', 'firms_wealth', 'firms_profit', 'gini_index',
                  'average_utility', 'inflation', 'average_qli']
    path = r'\\storage4\carga\MODELO DINAMICO DE SIMULACAO\Exits_python\JULY\SENSItivity\distributions'
    target1 = 'gdp_index'
    target2 = 'gini_index'
    a, b = get_data(path, target1, target2)
    main(a, b)
