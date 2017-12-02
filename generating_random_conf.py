import random
import pandas as pd

d_uniform = {"HOUSE_VACANCY": [0.02, 0.4], "MARKUP": [0, .5], "MEMBERS_PER_FAMILY": [1, 5],
             "PRODUCTION_MAGNITUDE": [1, 200], "SIZE_MARKET": [5, 50]}
d_taxes = {"TAX_CONSUMPTION": [.004, .00004], "TAX_ESTATE_TRANSACTION": [7e-06, 7e-08],
           "TAX_FIRM": [.004, .00004], "TAX_LABOR": [.0015, .000015], "TAX_PROPERTY": [4e-06, 4e-08],
           "TREASURE_INTO_SERVICES": [.8, 1.2]}
d_bool = ["ALTERNATIVE0", "FPM_DISTRIBUTION", "WAGE_IGNORE_UNEMPLOYMENT"]
d_perc = ["ALPHA", "BETA", "LABOR_MARKET", "PCT_DISTANCE_HIRING", "PERCENTAGE_ACTUAL_POP", "STICKY_PRICES"]
d_acps = ["MANAUS", "BELEM", "MACAPA", "SAO LUIS", "TERESINA", "FORTALEZA", "JUAZEIRO DO NORTE - CRATO - BARBALHA",
          "NATAL", "JOAO PESSOA", "CAMPINA GRANDE", "RECIFE", "PETROLINA - JUAZEIRO", "MACEIO", "ARACAJU",
          "SALVADOR", "FEIRA DE SANTANA", "ILHEUS - ITABUNA", "BELO HORIZONTE", "JUIZ DE FORA", "IPATINGA",
          "UBERLANDIA", "VITORIA", "VOLTA REDONDA - BARRA MANSA", "RIO DE JANEIRO", "CAMPOS DOS GOYTACAZES",
          "SAO PAULO", "CAMPINAS", "SOROCABA", "SAO JOSE DO RIO PRETO", "SANTOS", "JUNDIAI",
          "SAO JOSE DOS CAMPOS", "RIBEIRAO PRETO", "CURITIBA" "LONDRINA", "MARINGA", "JOINVILLE", "FLORIANOPOLIS",
          "PORTO ALEGRE", "NOVO HAMBURGO - SAO LEOPOLDO", "CAXIAS DO SUL", "PELOTAS - RIO GRANDE",
          "CAMPO GRANDE", "CUIABA", "GOIANIA", "BRASILIA"]


def generate(i=0):
    data = dict()
    for key in d_uniform.keys():
        data[key] = round(random.uniform(d_uniform[key][0], d_uniform[key][1]), 2)
    for key in d_taxes.keys():
        data[key] = round(random.uniform(d_taxes[key][0], d_taxes[key][1]), 8)
    for each in d_bool:
        data[each] = random.choice([True, False])
    for each in d_perc:
        data[each] = round(random.random(), 2)
    data['PROCESSSING_ACPS'] = random.choice(d_acps)
    return pd.DataFrame(data, index=[i])


def compound(n=10, df=pd.DataFrame()):
    for i in range(n):
        df = pd.concat([df, generate(i)])
    return df


if __name__ == '__main__':
    d = compound()
    print(d.describe())