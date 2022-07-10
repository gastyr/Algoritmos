from sklearn.linear_model import LinearRegression
from pandas_datareader import data as web
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as st
from bestfit import best_fit_distribution, remove_adjacent, CVaR, markov_plot, make_pdf

# declara a semente aleatoria
seed = 1337
# cria o "gerador" de numeros aleatorios baseado na semente aleatoria (container para o gerador de bits)
np_random_gen = np.random.Generator(np.random.PCG64(seed))

# retorna um valor entre 0 e 1, distribuicao uniforme
def rand01():
    return np_random_gen.uniform()

tickers = []

# nome do arquivo da carteira
wallet_name = 'wallet1'

#abre o arquivo contendo os ativos no formato leitura e adiciona na lista
with open(f'{wallet_name}.txt', 'r') as wallet:
    for line in wallet:
        tickers.append(line.rstrip())

# cria o dataframe dos ativos
data = pd.DataFrame()

# declara o intervalo de tempo para cálculo dos betas da reg. linear
start_beta = "01/01/2018"
end_beta = "01/01/2020"

# realiza o download das cotações
for t in tickers:
    cotacao = web.DataReader(f"{t}", data_source="yahoo", start=start_beta, end=end_beta)
    data[t] = cotacao["Adj Close"]


## baixa as cotações na janela de crise ##

# cria o dataframe da janela de crise
crisis_window = pd.DataFrame()

# declara o intervalo de tempo da crise test
start_crisis_test = "01/01/2020"
end_crisis_test = "01/01/2021"

# realiza o download das cotacoes no intervalo de crise
for t in tickers:
    cotacao = web.DataReader(f"{t}", data_source="yahoo", start=start_crisis_test, end=end_crisis_test)
    crisis_window[t] = cotacao["Adj Close"]

# transforma a cotacao no tempo de crise em retorno
crisis_test_return = np.log(crisis_window/crisis_window.shift())

#################################################################
# realiza o download do indice no intervalo de crise treino
start_crisis_train = "01/01/2008"
end_crisis_train = "01/01/2009"
index_crisis_train = web.DataReader('^BVSP', data_source="yahoo", start=start_crisis_train, end=end_crisis_train)

# calcula o retorno do indice no intervalo de crise treino
return_crisis_train = np.log(index_crisis_train["Adj Close"]/index_crisis_train["Adj Close"].shift())

# encontra a distribuição e parametros que melhor representam os dados empiricos do indice no intervalo de crise treino
fit = return_crisis_train.iloc[1:]
bd = best_fit_distribution(fit)
print(bd)
# manipula a distribuição e parametros para criar as amostras aleatórias
dist_name = bd[0:bd.find("(")]
params = bd[bd.find("(")+1:bd.find(")")]

# cria os dados para plotar a distribuicao teorica e o histograma empirico
dist_x , dist_y = make_pdf(bd)
hist,bins = np.histogram(fit, bins=50, density=True)

# plota o histograma dos retornos dos dados empiricos
histo = go.Figure()
histo.add_trace(go.Bar(x=bins, y=hist, name="Dados empíricos"))
histo.add_trace(go.Scatter(x=dist_x, y=dist_y,line = dict(color='rgb(55, 83, 109)', width=2), name="Distribuição teórica"))
histo.update_traces(marker_color='rgba(158,202,225,0.6)', marker_line_color='rgba(8,48,107,0.6)',
                  marker_line_width=1)
histo.update_layout(template='none', bargap=0, title_text=f"""Retorno do Ibovespa no intervalo de crise de {start_crisis_train} a {end_crisis_train}<br>"""
                                                                f"""Distribuição teórica mais ajustada {dist_name}({params})""")
histo.show()
#################################################################

# cria o dataframe dos coeficientes
tickers.remove('^BVSP')
coeff = pd.DataFrame(index = tickers)
# adiciona os pesos de cada ativo na carteira
print(f'tickers: {len(tickers)}')
coeff['weights'] = 1 / len(tickers)

#########################################################
# transformando a cotacao dos ativos em retorno
# calcula o retorno através do ln da divisão data[i]/data[i-1](normalização do preço dos ativos) => based in continuous compounding
log_returns = np.log(data/data.shift())


#########################################################
# criando a vetor X para o cálculo da regressão linear
X = log_returns['^BVSP'].iloc[1:].to_numpy().reshape(-1, 1)


#########################################################
#função que realiza a regressão linear
def reglin(ativo):
    print(ativo)
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

# busca a distribuicao encontrada na analise parametrica
scipy_random_gen = getattr(st, dist_name)
# usa o mesmo gerador(semente) numpy para gerar as variaveis da analise parametrica
scipy_random_gen.random_state = np_random_gen
# compila a string, evitando que passe pelo parser por cada chamada de funcao
code_rand = compile(f"scipy_random_gen.rvs({params})", "<string>", "eval")
# gera a variavel aleatoria com distribuicao da analise parametrica
def rand():
    return eval(code_rand)

# compila a string, evitando que passe pelo parser por cada chamada de funcao
code_pdf = compile(f"scipy_random_gen.pdf(x, {params})", "<string>", "eval")
# gera o numero aleatorio q(x) dado o valor de x ## funcao densidade de probabilidade pdf(x)
def pdf(x):
    return eval(code_pdf, {"scipy_random_gen": scipy_random_gen}, {"x": x})


# Monte Carlo
# número de simulações
n_sims = 100

# timeframe em dias
T = 252


# calcula o preco do início do portifolio: ultima cotacao do portfolio * contratos
initial_portfolio = np.inner(coeff.loc[:,'shares'], data.iloc[-1, 1:].values)

# valor do inicio da Cadeia de Markov
start = 0.1

# função que calcula o CAPM
def f(coeff, var):
    # transforma o retorno em um vetor de retornos, um valor para cada ativo no portfolio
    val_sim = np.full(shape=(len(tickers)), fill_value=var)
    # realiza o calcudo do CAPM: retorno = beta0 + beta1 * retorno_simulado
    retorno_diario = coeff.loc[:,'beta_0'].values + np.multiply(coeff.loc[:,'beta_1'].values, val_sim)
    # retorna a soma ponderada do retorno com os pesos
    #return abs(np.inner(coeff.loc[:,'weights'].values, retorno_diario))
    return np.inner(coeff.loc[:,'weights'].values, retorno_diario) + 1

# cria a classe para armazenar os candidados das simulacoes
class Candidatos():
    def __init__(self, start):
        self.aceitos = np.array([start])
        self.rejeitados = np.array([0])
        self.todos = np.array([start])

# cria a lista que armazena os objetos dos candidatos das simulacoes
simulation = []
for n in range(n_sims):
    # adicionando o objeto (que armazena os candidatos) na lista
    simulation.append(Candidatos(start))

# executa a simulacao MCMC
for n in range(n_sims):
    print(f'Executando simulação {n+1}')
    accept = 1
    while accept <= T:
        # gera as amostras aleatórias com a distribuição encontrada no teste de aderencia
        retorno_simulado = rand()
        # adiciona o candidato gerado no array de todos candidatos da simulacao
        simulation[n].todos = np.append(simulation[n].todos, retorno_simulado)
        # realiza o cálculo baseado no CAPM e a variavel aleatoria
        retorno_dia_ponderado = f(coeff, retorno_simulado) * pdf(simulation[n].aceitos[-1]) # pdf(x-1)
        # razao de metropolis, (f(x) * q(x-1)) / (f(x-1) * q(x))
        razao = retorno_dia_ponderado / (f(coeff, simulation[n].aceitos[-1]) * pdf(retorno_simulado)) #pdf(x)
        alpha = min(1, razao)
        if rand01() < alpha:
            accept += 1
            # adiciona a variavel aleatoria ao array dos candidatos aceitos
            simulation[n].aceitos = np.append(simulation[n].aceitos, retorno_simulado)
            # repete o ultimo candidato rejeitado ao array dos rejeitados
            simulation[n].rejeitados = np.append(simulation[n].rejeitados, simulation[n].rejeitados[-1])
            
        else:
            # repete o ultimo candidato aceito ao array dos aceitos
            simulation[n].aceitos = np.append(simulation[n].aceitos, simulation[n].aceitos[-1])
            # adiciona a variavel aleatoria ao array dos candidatos rejeitados
            simulation[n].rejeitados = np.append(simulation[n].rejeitados, retorno_simulado)


###################################################
# histograma dos retornos aceitos
return_hist = go.Figure()
return_hist.add_trace(go.Scatter(x=dist_x, y=dist_y,line = dict(color='rgb(55, 83, 109)', width=2), name="Distribuição teórica"))
minn = []
maxx = []
for n in range(n_sims):
    minn.append(np.min(simulation[n].aceitos[1:]))
    maxx.append(np.max(simulation[n].aceitos[1:]))
hist_min = np.min(minn)
hist_max = np.max(maxx)
for n in range(n_sims):
    hist_return,bins_edge = np.histogram(remove_adjacent(simulation[n].aceitos[1:]), bins=50, density=True,
                                                            range=(hist_min,hist_max))
    return_hist.add_trace(go.Bar(x=bins_edge, y=hist_return, name=f"{n+1}", opacity=0.5))
return_hist.update_layout(barmode='overlay', bargap=0)
return_hist.show()


# Calcula o CVaR das simulacoes e adiciona em uma lista
CVaR_list = []
for n in range(n_sims):
    CVaR_list.append(CVaR(simulation[n].aceitos[1:]))
# exporta a lista de CVaRs em um arquivo .txt
np.savetxt(f'{wallet_name}_CVaR.txt', CVaR_list, fmt='%4.8f')


###################################################
#calcula o CVaR histórico e plota
wallet_crisis = crisis_test_return.drop('^BVSP', axis=1)
wallet_crisis = wallet_crisis.dropna()
for t in tickers:
    wallet_crisis[t] = wallet_crisis[t] * coeff.loc[t,'weights']
wallet_crisis['row_sum'] = wallet_crisis.sum(axis=1)
historic_CVaR = CVaR(wallet_crisis['row_sum'])

cvar_plot = go.Figure()
cvar_plot.add_trace(go.Box(name=f'{wallet_name}',y=CVaR_list, showlegend=True,
                        jitter=0.5, boxmean='sd', boxpoints='all', notched=True, fillcolor='rgba(127, 96, 0, 0.5)'))
cvar_plot.add_hline(y=historic_CVaR, line_width=1.5, annotation_text=f'Historic CVaR {start_crisis_test} - {end_crisis_test}',
                annotation=dict(font_size=16), annotation_position="top left")
cvar_plot.update_layout(template='none', boxgap=0.8, title=f"CVaR histórico: {historic_CVaR * 100:.2f}%")              
cvar_plot.show()

accept_rate = []
for n in range(n_sims):
    accept_rate.append(len(remove_adjacent(simulation[n].aceitos[1:])) / len(simulation[n].todos[1:]))
print(f'Taxa de aceitação média: {np.mean(accept_rate) * 100:.2f}%')


###################################################
# criacao do grafico da evolucao do preco
fig = go.Figure()
days = np.arange(0, T+1)
plotly_colors = px.colors.qualitative.Plotly
count = int(np.ceil(n_sims / len(plotly_colors)))
colors = []
for i in range(count):
    colors.extend(px.colors.qualitative.Plotly)

for n,color in zip(range(n_sims),colors):
    simulation[n].aceitos[0] = 0
    price = np.cumprod(remove_adjacent(simulation[n].aceitos) + 1) * initial_portfolio
    fig.add_trace(go.Scatter(y=price, mode='lines', name=f'{n+1}', line_color=color))

fig.update_layout(template='none',title="Evolução do preço no tempo",
                    xaxis_title="Dias",
                    yaxis_title="Valor",
                    legend_title="Simulações",
                    legend_itemclick="toggleothers",
                    legend_itemsizing="constant")

fig.show()

#markov_plot(simulation, n=3)