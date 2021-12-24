from sklearn.linear_model import LinearRegression
from pandas_datareader import data as web
import pandas as pd
import numpy as np
import fitter as ft
import seaborn as sns

tickers= []
with open('wallet.txt', 'r') as wallet: #abre o arquivo contendo os ativos no formato leitura
    for line in wallet:
        tickers.append(line.rstrip())

data = pd.DataFrame()

start = "01/01/2020"
end = "01/01/2021"
beta_carteira = 0

for t in tickers:
    cotacao = web.DataReader(f"{t}", data_source="yahoo", start=start, end=end) #download das cotações
    data[t] = cotacao["Adj Close"]

log_returns = np.log(data/data.shift())
X = log_returns['^BVSP'].iloc[1:].to_numpy().reshape(-1, 1)

#########################################################
def beta(ativo):  # funcão para calcular o beta do ativo
    cov = log_returns.cov()
    var = log_returns['^BVSP'].var()
    
    beta = cov.loc[ativo, '^BVSP']/var

    return beta
#########################################################

def reglin(ativo): #função que realiza a regressão linear
    y = log_returns[ativo].iloc[1:].to_numpy().reshape(-1, 1)
    regr = LinearRegression()
    regr.fit(X, y)
    beta = regr.coef_[0, 0]
    print(f'Beta: {beta}')
    print(f"R2: {regr.score(X, y)}")
    beta_ponderado = beta * 0.1
    return beta_ponderado
    #print(f'Beta: {beta(ativo)}')

for t in tickers[1:]:   
    print(f"\n{t}")
    beta_carteira = beta_carteira + reglin(t)

print(f"\nBeta da carteira: {beta_carteira}")

fit = ft.Fitter(X,
           distributions=['norm',
                          'cauchy', 
                          "binom", 
                          "lognorm", 
                          "pareto",
                          "poisson",
                          "expon",
                          "dweibull",
                          "gamma",
                          "chi2",
                          "t",
                          "beta",
                          "triang",
                          "hypsecant"])

fit.fit()

print(fit.get_best(method = 'sumsquare_error'))
'''
futuro = np.random.normal(loc=0, scale=10, size=100) * beta_carteira


def VaR(retorno, alpha=5):
    return np.percentile(retorno, alpha)


def CVaR(returns, alpha=5):
    belowVaR = returns <= VaR(returns, alpha=alpha)
    return returns[belowVaR].mean()

print(f"CVaR: {CVaR(futuro)}")
print(f"VaR: {VaR(futuro)}")
'''