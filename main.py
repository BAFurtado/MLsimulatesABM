import os

import pandas as pd
from numpy import set_printoptions

import descriptive_stats
import generating_random_conf
import machines
import preparing_data
from sklearn.model_selection import train_test_split
set_printoptions(precision=4)


def get_data(pathways, col1, col2):
    name = 'pre_processed_data\\' + pathways[-4:] + '_x.csv'
    if os.path.exists(name):
        x = pd.read_csv(name, sep=';', index_col=False).drop('Unnamed: 0', axis=1)
        y = pd.read_csv(name.replace('x', 'y'), sep=';', index_col=False)
        print('Loaded!')
    else:
        x, y = preparing_data.main(pathways, col1, col2)
        x.to_csv(name, sep=';')
        y.to_csv(name.replace('x', 'y'), sep=';')
        print('Saved!')
    return train_test_split(x, y['target'], test_size=0.2, random_state=10)


def main(x, xt, y, yt):

    # Running model
    m2 = machines.run_random_forest_split(x, xt, y, yt)

    # Generating random configuration data to test against optimal results
    r = generating_random_conf.compound()
    print('Generated dataset summary')

    # Predicting results using machine on generated set of random parameters
    y2 = machines.predict(m2, r[x.columns.tolist()])
    y2 = pd.DataFrame({'name_model': y2.tolist()})

    # Output basic descriptive stats
    # Sending over X and Y as lists in a dictionary for actual and each model
    descriptive_stats.print_conf_stats({'actual': [pd.concat([x, xt], axis=0), pd.concat([y, yt], axis=0)],
                                        'random_forest': [r, y2]})


if __name__ == "__main__":
    cols_names = ['months', 'price_index', 'gdp_index', 'gdp_growth', 'unemployment', 'average_workers',
                  'families_wealth', 'families_savings', 'firms_wealth', 'firms_profit', 'gini_index',
                  'average_utility', 'inflation', 'average_qli']
    path = r'\\storage4\carga\MODELO DINAMICO DE SIMULACAO\Exits_python\JULY'
    target1 = 'gdp_index'
    target2 = 'gini_index'
    x_train, x_test, y_train, y_test = get_data(path, target1, target2)
    main(x_train, x_test, y_train, y_test)
