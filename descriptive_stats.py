import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format


def print_conf_stats(kwargs):
    for i, arg in enumerate(kwargs['bases']):
        df = arg.describe().T
        df.drop(['25%', '75%'], axis=1, inplace=True)
        print(df.head())
        name = 'outputs\\' + kwargs['text'][i] + '.csv'
        with open(name, 'wb'):
            df.to_csv(name, sep=';', float_format='%.6f')
