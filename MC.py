from sklearn.linear_model import LinearRegression
from pandas_datareader import data as web
import pandas as pd
import numpy as np
import plotly.express as px
import scipy.stats as stats
from bestfit import best_fit_distribution


def rand01():
    return np.random.uniform()

tickers= []
#abre o arquivo contendo os ativos no formato leitura e adiciona na lista
with open('wallet.txt', 'r') as wallet:
    for line in wallet:
        tickers.append(line.rstrip())

# cria o dataframe dos ativos
data = pd.DataFrame()

# declara o intervalo de tempo
start = "01/01/2020"
end = "01/01/2021"

# realiza o download das cotações
for t in tickers:
    cotacao = web.DataReader(f"{t}", data_source="yahoo", start=start, end=end)
    data[t] = cotacao["Adj Close"]


# cria o dataframe dos coeficientes
tickers.remove('^BVSP')
coeff = pd.DataFrame(index = tickers)
# adiciona os pesos de cada ativo na carteira
coeff['weights'] = 0.1

#########################################################
# transformando a cotação dos ativos em retorno/ 3 formas de fazer

# calcula o retorno através do logaritmo e normalização do preço dos ativos
log_returns = np.log(data/data.shift())

# calcula as variações diárias (retorno), usando pandas
returns = data.pct_change()

# calcula o log do retorno
lr = np.log(1 + data.pct_change())

#########################################################
# criando a vetor X para o cálculo da regressão linear
X = log_returns['^BVSP'].iloc[1:].to_numpy().reshape(-1, 1)

# comparando os 3 metodos de transformar o preço em retorno
#figure = pd.DataFrame()
#figure['log_returns'] = log_returns['^BVSP']
#figure['returns'] = returns['^BVSP']
#figure['lr'] = lr['^BVSP']
#figure['diff'] = np.log(data['^BVSP'].diff())
#fig2 = px.line(figure)
#fig2.show()

#########################################################
#função que realiza a regressão linear
def reglin(ativo):
    y = log_returns[ativo].iloc[1:].to_numpy().reshape(-1, 1)
    regr = LinearRegression()
    regr.fit(X, y)
    beta_0 = regr.intercept_
    beta_1 = regr.coef_[0, 0]
    R2 = regr.score(X, y)
    return beta_0, beta_1, R2

#########################################################
# calcula os coeficientes de cada ativo e adiciona no dataframe
for t in tickers:   
    beta0, beta1, R2 = reglin(t)
    coeff.at[t, 'beta_0'] = beta0
    coeff.at[t, 'beta_1'] = beta1
    coeff.at[t, 'R2'] = R2

# valor inicial do portfolio
initial_portfolio = 10000
# calcula a quantidade de papeis de cada ativo (valor total * peso / cotacao)
cotas = (initial_portfolio * coeff.loc[:,'weights']) / data.iloc[-1, 1:]
# adiciona o valor inteiro dos papeis nos coeficientes
coeff.at[:, 'shares'] = cotas.astype(int)


# encontra a distribuição e parametros que melhor representam os dados empíricos
fit = pd.Series(log_returns['^BVSP'].iloc[1:])
bd = best_fit_distribution(fit)
print(bd)
# manipula a distribuição e parametros para criar as amostras aleatórias
name = bd[0:bd.find("(")]
params = bd[bd.find("("):bd.find(")")]

def rand():
    return eval(f"stats.{name}.rvs{params})")

# Monte Carlo
# número de simulações
n_sims = 100

# timeframe em dias
T = 252

# simulações da carteira
markov_chain = np.full(shape=(T+1, n_sims), fill_value=0)

# calcula o início da cadeia de Markov: ultima cotacao do portfolio * contratos
initial_portfolio = np.inner(coeff.loc[:,'shares'], data.iloc[-1, 1:].values)
start = 0.5

# função que calcula o CAPM
def f(coeff, var):
    sim = np.full(shape=(len(tickers)), fill_value=var)
    # realiza o calcudo do CAPM: retorno = beta0 + beta1 * retorno_simulado
    retorno_diario = coeff.loc[:,'beta_0'].values + np.multiply(coeff.loc[:,'beta_1'].values, sim)
    # retorna a soma ponderada do retorno com os pesos
    return np.inner(coeff.loc[:,'weights'].values, retorno_diario)


return_list = [[] for i in range(n_sims)]
return_accept = [[] for i in range(n_sims)]
reject = 0

for n in range(n_sims):
    #del price_list = []
    return_list[n].append(start)
    return_accept[n].append(start)
    accept = 1
    while accept <= T:
        # gera as amostras aleatórias com a distribuição encontrada no teste de aderencia
        retorno_simulado = rand()
        retorno_dia_ponderado = f(coeff, retorno_simulado)
        razao = retorno_dia_ponderado / f(coeff, return_list[n][-1])
        alpha = min(1, abs(razao))
        if rand01() < alpha:
            accept += 1
            return_accept[n].append(retorno_dia_ponderado)
            return_list[n].append(retorno_dia_ponderado)
        else:
            reject += 1
    markov_chain[:,n] = np.cumprod(np.array(return_accept[n]) + 1) * initial_portfolio



print(f'Taxa de aceitação: {(T*n_sims)/((T*n_sims) + reject)}')
print(markov_chain)

# função que calcula o VaR
def mcVaR(returns, alpha=5):
    """ Input: pandas series of returns
        Output: percentile on return distribution to a given confidence level alpha
    """
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    else:
        raise TypeError("Expected a pandas data series.")

# função que calcula o CVaR
def mcCVaR(returns, alpha=5):
    """ Input: pandas series of returns
        Output: CVaR or Expected Shortfall to a given confidence level alpha
    """
    if isinstance(returns, pd.Series):
        belowVaR = returns <= mcVaR(returns, alpha=alpha)
        return returns[belowVaR].mean()
    else:
        raise TypeError("Expected a pandas data series.")


portResults = pd.Series(markov_chain[-1,:])

var = mcVaR(portResults, alpha=5)
VaR = initial_portfolio - var
cvar = mcCVaR(portResults, alpha=5)
CVaR = initial_portfolio - cvar

print(f'VaR R${VaR:.2f}')
print(f'CVaR R${CVaR:.2f}')

# criação do gráfico
fig = px.line(markov_chain)
fig.add_hline(y=var, line_width=2, line_dash="dot", annotation_text='VaR',
                annotation=dict(font_size=16), annotation_position="top left")

fig.add_hline(y=cvar, line_width=2, line_dash="dash", annotation_text='CVaR',
                annotation=dict(font_size=16), annotation_position="bottom left")
                
fig.update_layout(template='none',title="Evolução do preço no tempo",
                    xaxis_title="Dias",
                    yaxis_title="Valor",
                    legend_title="Simulações",
                    legend_itemclick="toggleothers",
                    legend_itemsizing="constant")

fig.show()
