import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format


def print_conf_stats(*args, **kwargs):
    for arg in args:
        df = arg.describe().T
        df.drop(['25%', '75%'], axis=1, inplace=True)
        print(df)
        with open(arg + '.csv', 'wb') as f:
            f.write(df)
