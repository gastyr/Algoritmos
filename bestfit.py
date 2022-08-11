import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats._continuous_distns import _distn_names
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import itertools
# method: sum of squared estimate of errors (SSE)
# Create models from data
def best_fit_distribution(data, bins=50):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Best holders
    best_distributions = []

    # Estimate distribution parameters from data
    for ii, distribution in enumerate([d for d in _distn_names if not d in ['levy_stable', 'studentized_range']]):

        print("{:>3} / {:<3}: {}".format( ii+1, len(_distn_names), distribution ))

        distribution = getattr(st, distribution)

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                
                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                
                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))
                

                # identify if this distribution is better
                best_distributions.append((distribution, params, sse))
        
        except Exception:
            pass

    best_distributions = sorted(best_distributions, key=lambda x:x[2])
    best_dist = best_distributions[0]
    param_names = (best_dist[0].shapes + ', loc, scale').split(', ') if best_dist[0].shapes else ['loc', 'scale']
    param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_dist[1])])
    dist_str = '{}({}){}'.format(best_dist[0].name, param_str,best_dist[2])
    return dist_str

def make_pdf(dist, size=1000):
    # nome da distribuicao
    name = dist[0:dist.find("(")]
    # parametros da distribuicao
    params = dist[dist.find("(")+1:dist.find(")")]
    
    # pega os pontos de inicio e fim
    start = eval(f'st.{name}.ppf(0.01, {params})')
    end = eval(f'st.{name}.ppf(0.99, {params})')

    # discretiza o eixo x e cria a funcao de densidade de probabilidade
    x = np.linspace(start, end, size)
    string = f'st.{name}.pdf(x, {params})'
    y = eval(string)

    return x, y

def remove_adjacent(iter):
    list = [k for k, g in itertools.groupby(iter)]
    return np.array(list)  

def CVaR(series, alpha = 0.05):
    sorted = np.sort(remove_adjacent(series))
    count = len(sorted)
    location = round(count * alpha)
    VaR = sorted[location - 1]
    CVaR = np.mean(sorted[sorted < VaR])
    return CVaR
    
######## Grafico ########
def markov_plot(candidatos, n=5):
    colors = ['#3f3f3f', '#00bfff', '#ff7f00']

    length = len(candidatos)
    chosen = np.linspace(0, length-1, n).astype(int)

    for i in chosen:
        fig = make_subplots(
            rows=3, cols=2,
            column_widths=[0.58, 0.42],
            row_heights=[1., 1., 1.],
            specs=[[{"type": "scatter"}, {"type": "xy"}],
                [{"type": "scatter"}, {"type": "xy", "rowspan": 2}],
                [{"type": "scatter"},            None           ]])

        fig.add_trace(
            go.Scatter(x = np.arange(1, len(candidatos[i].aceitos)+1, 1), 
                        y = candidatos[i].aceitos,
                        hoverinfo = 'x+y',
                        mode='lines',
                        line=dict(color='#3f3f3f',
                        width=1),
                        showlegend=False,
                        ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x = np.arange(1, len(candidatos[i].rejeitados)+1, 1), 
                        y = candidatos[i].rejeitados,
                        hoverinfo = 'x+y',
                        mode='lines',
                        line=dict(color='#00bfff',
                        width=1),
                        showlegend=False,
                        ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x = np.arange(1, len(candidatos[i].todos)+1, 1), 
                        y = candidatos[i].todos,
                        hoverinfo = 'x+y',
                        mode='lines',
                        line=dict(color='#ff7f00',
                        width=1),
                        showlegend=False,
                        ),
            row=3, col=1
        )

        boxfig= go.Figure(data=[go.Box(x=candidatos[i].aceitos, showlegend=False, notched=True, marker_color="#3f3f3f", name='aceitos'),
                                go.Box(x=candidatos[i].rejeitados, showlegend=False, notched=True, marker_color="#00bfff", name='rejeitados'),
                                go.Box(x=candidatos[i].todos, showlegend=False, notched=True, marker_color="#ff7f00", name='todos')])

        for k in range(len(boxfig.data)):
            fig.add_trace(boxfig.data[k], row=1, col=2)

        group_labels = ['\u03B8 aceitos', '\u03B8 rejeitados', 'Todos os \u03B8']
        hist_data = [candidatos[i].aceitos, candidatos[i].rejeitados, candidatos[i].todos]

        distplfig = ff.create_distplot(hist_data, group_labels, colors=colors,
                                show_rug=False, bin_size=.01)

        for k in range(len(distplfig.data)):
            fig.add_trace(distplfig.data[k],
            row=2, col=2
        )
        acc_rate = len(remove_adjacent(candidatos[i].aceitos)) / len(remove_adjacent(candidatos[i].todos)) * 100
        fig.update_layout(barmode='overlay', template='none', title=f"Cadeia de Markov da simulação {i+1}. Taxa de aceitação: {acc_rate:.2f}%")
        fig.show()

######## Grafico da cadeia de Markov ########
def markovchain_plot(wallet_name, candidatos, n=5):
    color = '#3f3f3f'

    length = len(candidatos)
    chosen = np.linspace(0, length-1, n).astype(int)

    for i in chosen:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = np.arange(1, len(candidatos[i].aceitos)+1, 1), 
                        y = candidatos[i].aceitos,
                        hoverinfo = 'x+y',
                        mode='lines',
                        line=dict(color=color,
                        width=1),
                        showlegend=False,
                        ))
        acc_rate = len(remove_adjacent(candidatos[i].aceitos)) / len(remove_adjacent(candidatos[i].todos)) * 100
        fig.update_layout(template='none', width=1200, height=800,
                            title=f"Cadeia de Markov da simulação {i+1}. Taxa de aceitação: {acc_rate:.2f}%, "
                            f"{len(remove_adjacent(candidatos[i].todos))} candidatos propostos.")
        fig.write_image(f"imagens/mcmc_simulacao{i+1}_{wallet_name}.png")
        fig.show()


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