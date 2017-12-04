import os
import pickle

import pandas as pd
from numpy import set_printoptions

import descriptive_stats
import generating_random_conf
import machines
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


def print_results(conf, y):
    res = pd.concat([conf, y], axis=1)
    res = res.groupby(by=['t2']).agg('mean')
    res.T.to_csv('outputs\\comparison.csv', sep=';')
    print('\n Final results: \n')
    print(res)


def main(x, y):
    print('\n Training dataset summary: \n')
    descriptive_stats.print_conf_stats({'bases': [x], 'text': ['actual']})
    m2 = machines.run_random_forest_split(x, y)
    r = generating_random_conf.compound()
    print('Generated dataset summary')
    descriptive_stats.print_conf_stats({'bases': [r], 'text':['generated']})
    y2 = machines.predict_random_forest_cross(m2, r[x.columns.tolist()])
    y2 = pd.DataFrame({'t2': y2.tolist()})
    print_results(r, y2)


if __name__ == "__main__":
    cols_names = ['months', 'price_index', 'gdp_index', 'gdp_growth', 'unemployment', 'average_workers',
                  'families_wealth', 'families_savings', 'firms_wealth', 'firms_profit', 'gini_index',
                  'average_utility', 'inflation', 'average_qli']
    path = r'\\storage4\carga\MODELO DINAMICO DE SIMULACAO\Exits_python\JULY'
    target1 = 'gdp_index'
    target2 = 'gini_index'
    a, b = get_data(path, target1, target2)
    main(a, b)
