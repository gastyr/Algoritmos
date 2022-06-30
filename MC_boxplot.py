import numpy as np
import plotly.graph_objects as go

wallet1 = np.loadtxt('wallet1_CVaR.txt')
wallet2 = np.loadtxt('wallet2_CVaR.txt')
wallet3 = np.loadtxt('wallet3_CVaR.txt')

fig = go.Figure()

fig.add_trace(go.Box(y=wallet1, showlegend=True, jitter=0.5, boxmean='sd', boxpoints='all', notched=True, marker_color="#ffaa00", name='Blue Chips'))

fig.add_trace(go.Box(y=wallet2, showlegend=True, jitter=0.5, boxmean='sd', boxpoints='all', notched=True, marker_color="#ae2012", name='Small Caps'))

fig.add_trace(go.Box(y=wallet3, showlegend=True, jitter=0.5, boxmean='sd', boxpoints='all', notched=True, marker_color="#283618", name='Mista'))

print(f'Carteira 1: média:{np.mean(wallet1):.4f}, desvio padrão:{np.std(wallet1):.4f}, máximo:{np.max(wallet1):.4f} e mínimo:{np.min(wallet1):.4f}')
print(f'Carteira 2: média:{np.mean(wallet2):.4f}, desvio padrão:{np.std(wallet2):.4f}, máximo:{np.max(wallet2):.4f} e mínimo:{np.min(wallet2):.4f}')
print(f'Carteira 3: média:{np.mean(wallet3):.4f}, desvio padrão:{np.std(wallet3):.4f}, máximo:{np.max(wallet3):.4f} e mínimo:{np.min(wallet3):.4f}')

fig.update_layout(template='none', boxgap=0.7,
title=f"""Blue Chips: média:{np.mean(wallet1):.4f}, desvio padrão:{np.std(wallet1):.4f}, máximo:{np.max(wallet1):.4f} e mínimo:{np.min(wallet1):.4f}<br>"""
f"""Small Caps 2: média:{np.mean(wallet2):.4f}, desvio padrão:{np.std(wallet2):.4f}, máximo:{np.max(wallet2):.4f} e mínimo:{np.min(wallet2):.4f}<br>"""
f"""Mista 3: média:{np.mean(wallet3):.4f}, desvio padrão:{np.std(wallet3):.4f}, máximo:{np.max(wallet3):.4f} e mínimo:{np.min(wallet3):.4f}<br>"""
)
fig.show()