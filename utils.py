from __future__ import division

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from IPython.display import display, HTML

from tqdm import tqdm

import plotly.graph_objects as go

def closest(lst, K):
     
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]

def print_table(table,
                name=None,
                float_format=None,
                formatters=None,
                header_rows=None):
    """
    Pretty print a pandas DataFrame.
    Uses HTML output if running inside Jupyter Notebook, otherwise
    formatted text output.
    Parameters
    ----------
    table : pandas.Series or pandas.DataFrame
        Table to pretty-print.
    name : str, optional
        Table name to display in upper left corner.
    float_format : function, optional
        Formatter to use for displaying table elements, passed as the
        `float_format` arg to pd.Dataframe.to_html.
        E.g. `'{0:.2%}'.format` for displaying 100 as '100.00%'.
    formatters : list or dict, optional
        Formatters to use by column, passed as the `formatters` arg to
        pd.Dataframe.to_html.
    header_rows : dict, optional
        Extra rows to display at the top of the table.
    """

    if isinstance(table, pd.Series):
        table = pd.DataFrame(table)

    if name is not None:
        table.columns.name = name

    html = table.to_html(float_format=float_format, formatters=formatters)

    if header_rows is not None:
        # Count the number of columns for the text to span
        n_cols = html.split('<thead>')[1].split('</thead>')[0].count('<th>')

        # Generate the HTML for the extra rows
        rows = ''
        for name, value in header_rows.items():
            rows += ('\n    <tr style="text-align: right;"><th>%s</th>' +
                     '<td colspan=%d>%s</td></tr>') % (name, n_cols, value)

        # Inject the new HTML
        html = html.replace('<thead>', '<thead>' + rows)

    display(HTML(html))

def EfficiencyFrontier(daily_returns,
                       anual = True,
                       annualise = 252, 
                       n_assets = 3, 
                       n_portfolio = 10000, 
                       risk_free_rate = 0.0):

    """
EfficiencyFrontier plots the Efficiency Frontier based on Markowitz theory of portfolio diversification, with Sharpe Ratio for every company,
and gathers a chart for all the data concerning the portfolios in question, to then retrieve the best suited.

daily_returns: returns of the assets in the portfolio (DataFrame)
anual = True: if True, returns are annualised, default value False (bool)
annualise: number of trading days in a year, daily -> 252 / monthly -> 12, default value 252(int)
n_assets: number of assets in the portfolio, default value 3 (int)
n_portfolio: number of portfolios to be generated, default value 10000 (int)
risk_free_rate: risk free rate, default value 0.0 (float)
return:    port: dataframe with the portfolio information
            fig: plotly figure with the Efficiency Frontier
""" 



    #-- Get annualised mean returns
    mus = (1+daily_returns.mean())**annualise - 1
    # -- Or non-annualised mean returns
    mus2 = daily_returns.mean()

    #-- Get covariances and variances
    #- Variance along diagonal of covariance matrix
    #- Multiply by annualise to annualise it
    #- https://quant.stackexchange.com/questions/4753/annualized-covariance
    cov = daily_returns.cov()*annualise

    #-- Create random portfolio weights and indexes
    #- How many assests in the portfolio

    mean_variance_pairs = []
    weights_list=[]
    tickers_list=[]

    for i in tqdm(range(n_portfolio)):
        next_i = False
        while True:
            #- Choose assets randomly without replacement
            assets = np.random.choice(list(daily_returns.columns), n_assets, replace=False)
            #- Choose weights randomly ensuring they sum to one
            weights = np.random.rand(n_assets)
            weights = weights/sum(weights)

            #-- Loop over asset pairs and compute portfolio return and variance
            portfolio_E_Variance = 0
            portfolio_E_Return = 0
            if anual == True:
                for i in range(len(assets)):
                    portfolio_E_Return += weights[i] * mus.loc[assets[i]]
                    for j in range(len(assets)):
                        portfolio_E_Variance += weights[i] * weights[j] * cov.loc[assets[i], assets[j]]
            else:
                for i in range(len(assets)):
                    portfolio_E_Return += weights[i] * mus2.loc[assets[i]]
                    for j in range(len(assets)):
                        portfolio_E_Variance += weights[i] * weights[j] * cov.loc[assets[i], assets[j]]

            #-- Skip over dominated portfolios
            for R,V in mean_variance_pairs:
                if (R > portfolio_E_Return) & (V < portfolio_E_Variance):
                    next_i = True
                    break
            if next_i:
                break

            #-- Add the mean/variance pairs to a list for plotting
            mean_variance_pairs.append([portfolio_E_Return, portfolio_E_Variance])
            weights_list.append(weights)
            tickers_list.append(assets)
            break

    cata =pd.DataFrame( mean_variance_pairs, columns = ['Rentabilidad', 'Volatilidad'])
    port  = pd.DataFrame(tickers_list)
    weight  = pd.DataFrame(weights_list)
    for i in range (n_assets):
        port.rename(columns={i:'comp_'+str(i)}, inplace=True)
        weight.rename(columns={i:'peso_'+str(i)}, inplace=True)

    port = pd.concat([port, weight, cata], axis=1)
    port['Volatilidad'] = port['Volatilidad']**0.5
    port['sharpe'] =  (port['Rentabilidad']-risk_free_rate)/(port['Volatilidad'])
    port['instrument'] = ['portfolio_'+str(i) for i in range(1, len(port)+1)]
    port = port.set_index('instrument')
    print_table(port.head())

    #-- Plot the risk vs. return of randomly generated portfolios
    #-- Convert the list from before into an array for easy plotting
    mean_variance_pairs = np.array(mean_variance_pairs)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mean_variance_pairs[:,1]**0.5, y=mean_variance_pairs[:,0], 
                        marker=dict(color=(mean_variance_pairs[:,0]-risk_free_rate)/(mean_variance_pairs[:,1]**0.5), 
                                    showscale=True, 
                                    size=7,
                                    line=dict(width=1),
                                    colorscale="RdBu",
                                    colorbar=dict(title="Sharpe<br>Ratio")
                                    ), 
                        mode='markers',
                        text=[str(np.array(tickers_list[i])) + "<br>" + str(np.array(weights_list[i]).round(2)) for i in range(len(tickers_list))]))
    if anual == True:
        fig.update_layout(template='plotly_white',
                        xaxis=dict(title='Volatilidad Anualizada'),
                        yaxis=dict(title='Retorno Anualizado'),
                        title='Frontera de Eficiencia',
                        width=850,
                        height=500)
    else:
        fig.update_layout(template='plotly_white',
                        xaxis=dict(title='Volatilidad Menusal'),
                        yaxis=dict(title='Retorno Mensual'),
                        title='Frontera de Eficiencia',
                        width=850,
                        height=500)
    fig.update_xaxes(range=[0.0, 0.26])
    fig.update_yaxes(range=[-0.2,0.29])
    fig.update_layout(coloraxis_colorbar=dict(title="Sharpe Ratio"))
    fig.show()
    return port, fig