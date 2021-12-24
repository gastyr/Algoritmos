from sklearn.linear_model import LinearRegression
from pandas_datareader import data as web
import pandas as pd
import numpy as np

tickers= []
with open('wallet.txt', 'r') as wallet: #abre o arquivo contendo os ativos no formato leitura
    for line in wallet:
        tickers.append(line.rstrip())

data = pd.DataFrame()

start = "01/01/2020"
end = "11/10/2020"

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
    print(f'Beta: {regr.coef_[0, 0]}')
    #print(f'Beta: {beta(ativo)}')
    print(f"R2: {regr.score(X, y)}")

for t in tickers[1:]:
    print(f"\n{t}")
    reglin(t)