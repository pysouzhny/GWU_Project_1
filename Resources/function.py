# imports
import plotly.express as px
import panel as pn
import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
#%matplotlib inline
import panel as pn
pn.extension('plotly')
import hvplot.pandas
import seaborn as sns
import montecarlo as mc

from panel.interact import interact
import random
from iexfinance.stocks import get_historical_data
import iexfinance as iex
from ta import *
from dotenv import load_dotenv
import os
import panel as pn
import json
from urllib.request import Request, urlopen
#from data_collection import get_crypo_from_API

import warnings
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.2f}'.format

load_dotenv
api_key = os.environ.get("IEX_TOKEN")

def import_data_api(url):
    request = Request(url)
    response = urlopen(request)
    #print(response)
    data = response.read()
    #print(data)
    json_data = json.loads(data)
    return json_data

def get_raw_dataframe(ticker_list = [], allData=False,limit=360):
    
    load_dotenv
    api_key = os.environ.get("IEX_TOKEN")
    
    raw_df = pd.DataFrame()

    for ticker in ticker_list:

        load_dotenv
        api_key = os.environ.get("IEX_TOKEN")
        #if allData is true, then it gets all the data available. If not, select data according to limit.
        if allData:
            url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={ticker}&tsym=USD&allData=true&api_key={api_key}"
        else:
            url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={ticker}&tsym=USD&limit={limit}&api_key={api_key}"
        raw_data = import_data_api(url)
        df1 = pd.DataFrame(raw_data['Data']['Data'])
        df=df1
        df['time'] = pd.to_datetime(df1['time'],unit='s')
        df.set_index(df['time'], inplace=True)
        df.drop(['time'], axis=1, inplace=True)
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['volumefrom'] = df['volumefrom'].astype(float)
        df['volumeto'] = df['volumeto'].astype(float)

        df.columns = ticker + ' ' + df.columns
        df=df.filter(regex='close|high|low|volume')
        raw_df = pd.concat([raw_df, df], axis=1)


        #print(json.dumps(raw_data, indent=5))
        
    return raw_df

def get_raw_dataframe_trade(ticker_list = [], allData=False,limit=360):
    
    load_dotenv
    api_key = os.environ.get("IEX_TOKEN")
    
    raw_df = pd.DataFrame()

    for ticker in ticker_list:

        load_dotenv
        api_key = os.environ.get("IEX_TOKEN")
        #if allData is true, then it gets all the data available. If not, select data according to limit.
        if allData:
            url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={ticker}&tsym=USD&allData=true&api_key={api_key}"
        else:
            url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={ticker}&tsym=USD&limit={limit}&api_key={api_key}"
        raw_data = import_data_api(url)
        df1 = pd.DataFrame(raw_data['Data']['Data'])
        df=df1
        df['time'] = pd.to_datetime(df1['time'],unit='s')
        df['time'] = df['time'].astype(str)
        #df.set_index(df['time'], inplace=True)
        #df.drop(['time'], axis=1, inplace=True)

        #df.columns = ticker + ' ' + df.columns
        df=df.filter(regex='time|close|volume')
        df['ticker']=ticker
        df.drop(['volumefrom',], axis=1, inplace=True)
        df.rename(columns={'volumeto':'volume'}, inplace=True)
        df['Total Trade'] = df['close']*df['volume']
        raw_df = pd.concat([raw_df, df])


        #print(json.dumps(raw_data, indent=5))
        
    return raw_df



def get_raw_dataframe_row(ticker_list = [], allData=False,limit=360):
    
    load_dotenv
    api_key = os.environ.get("IEX_TOKEN")
    
    raw_df = pd.DataFrame()

    for ticker in ticker_list:

        load_dotenv
        api_key = os.environ.get("IEX_TOKEN")
        #if allData is true, then it gets all the data available. If not, select data according to limit.
        if allData:
            url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={ticker}&tsym=USD&allData=true&api_key={api_key}"
        else:
            url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={ticker}&tsym=USD&limit={limit}&api_key={api_key}"
        raw_data = import_data_api(url)
        df1 = pd.DataFrame(raw_data['Data']['Data'])
        df=df1
        df['time'] = pd.to_datetime(df1['time'],unit='s')
        df['time'] = df['time'].astype(str)
        #df.set_index(df['time'], inplace=True)
        #df.drop(['time'], axis=1, inplace=True)

        #df.columns = ticker + ' ' + df.columns
        df=df.filter(regex='time|volume')
        df['ticker']=ticker
        raw_df = pd.concat([raw_df, df])


        #print(json.dumps(raw_data, indent=5))
        
    return raw_df

  
def get_close_price_df(dataframe):

    daily_close_df=dataframe.filter(regex='close')

    return daily_close_df


def strd_dev(portfolio_df):
    portfolio_df.dropna(inplace=True)
    strd_deviation  = portfolio_df.pct_change()
    return strd_deviation

def corr_plot(portfolio_df):
    
    portfolio_daily_retn = strd_dev(portfolio_df)

    title_font = {'family': 'monospace',
            'color':  'blue',
            'weight': 'bold',
            'size': 15,
            }
    correlated = portfolio_daily_retn.corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(correlated, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    # Set up the matplotlib figure
    correlated_plot, ax = plt.subplots(figsize=(7,7))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(correlated, mask=mask, cmap="coolwarm", vmax=1, vmin =-1, center=0,
                square=True, linewidths=.5, annot=True
                #cbar_kws={"shrink": .5}
               )
    plt.title(f"Correlation Map of Portfolio\n",fontdict=title_font)
    ax.set_facecolor("aliceblue")
    
    #correlated_plot = sns.heatmap(correlated, vmin=-1, vmax=1, annot=True,cmap="coolwarm") 
    plt.close()
    return pn.Pane(correlated_plot)

def sharp_rt_plot(portfolio_df):

    portfolio_daily_retn = strd_dev(portfolio_df)
    
    title_font = {'family': 'monospace',
            'color':  'blue',
            'weight': 'bold',
            'size': 15,
            }
    label_font = {'family': 'monospace',
            'color':  'green',
            'weight': 'bold',
            'size': 12,
            }
   # bar_colors=["orange","plum","yellowgreen","indigo","wheat","salmon","lightblue","purple","gold",
    #           "cornflowerblue","mediumslateblue","seagreen","peru"]
    bar_colors=["midnightblue","royalblue","indigo","darkcyan","darkgreen","maroon",
               "purple","darkorange","slategray","forestgreen"]
    sharp_ratios = portfolio_daily_retn.mean()*np.sqrt(252)/portfolio_daily_retn.std()
    
    sr_plot = plt.figure();
    plt.bar(x = sharp_ratios.index, height=sharp_ratios, color=random.sample(bar_colors,len(sharp_ratios.index)))
    plt.title(f"Sharp Ratios of Portfolio\n",fontdict=title_font)
    plt.ylabel("Sharp Ratio",fontdict=label_font)
    plt.xlabel("Assets",fontdict=label_font)
    plt.axhline(sharp_ratios.mean(), color='r')
    plt.close()
    return pn.Pane(sr_plot)


  

def plot_mont_carl(portfolio_df, trial=50):

    plot_title = f"Monte-Carlo Simulation of Portfolio"
    mc_sim = mc.monte_carlo_sim(portfolio_df,trial)

    monte_carlo_sim_plot = mc_sim.hvplot(title='Monte-Carlo Simulation',figsize=(18,10),legend=False)
    return monte_carlo_sim_plot


##fonction cumul return
def cumul_plot(portfolio_daily):

    portfolio_daily_retn = strd_dev(portfolio_daily)

    cumul_rtn_plot = ( 1 + portfolio_daily_retn).cumprod().plot(figsize=(14, 10), title="Cumulative Returns")

    return cumul_rtn_plot

def drop_inf(portfolio_daily):
    dat = portfolio_daily
    dat = dat.replace([np.inf, -np.inf], np.nan).dropna()
    return dat

    
def distribution_plot(clean_dataframe):  

    clean_dataframe_rtn = clean_dataframe.pct_change().reset_index() 
    sns.set_theme(style="ticks", palette="pastel")

    t=sns.load_dataset("tips")

    Crypto_Daily_Returns_unpivoted=clean_dataframe_rtn.melt(id_vars='time',var_name='assets',value_name='daily return')
    Crypto_Daily_Returns_unpivoted.index=pd.DatetimeIndex(Crypto_Daily_Returns_unpivoted.index)
    Crypto_Daily_Returns_unpivoted.index.year

    sns.set(rc = {'figure.figsize':(20,10)})
    plot = sns.violinplot(x="assets", y="daily return",
            hue="assets", 
            data=Crypto_Daily_Returns_unpivoted)
    return plot



def get_scenario_portfolio(portfolio_df,num_portfolio=10000):

    #rename the columns to tickers
    portfolio_df.rename(columns={'ETH Close':'ETH','BTC Close':'BTC','DOGE Close':'DOGE','USDT Close':'USDT','SUSHI Close':'SUSHI'},inplace=True)

    #compute percent change and covariance matrix
    daily_returns=portfolio_df.pct_change().dropna()
    variance_matrix=len(daily_returns.index)*daily_returns.cov()
    #create empty list to store all returns, volatility and weights
    port_returns=[]
    port_volatility=[]
    port_weights=[]

    #find the number of assets to assign weight to
    num_assets=len(daily_returns.columns)


    #compute the expected return which is the mean of the retuns
    individual_returns=portfolio_df.pct_change().mean()*365#len(daily_returns.index)#df[(df.index=='2020-10-06')|(df.index=='2021-10-06')].pct_change().mean()*100

    #print the annualized returns
    individual_returns
    #we loop through each scenarios to find the weights, returns and volatility encountered
    for port in range(num_portfolio):
        weights=np.random.random(num_assets)
        weights=weights/np.sum(weights)
        port_weights.append(weights)
        returns= np.dot(weights,individual_returns)
        port_returns.append(returns)

        var=variance_matrix.mul(weights,axis=0).mul(weights,axis=1).sum().sum()
        sd=np.sqrt(var)

        ann_sd=sd*np.sqrt(len(daily_returns.index))
        port_volatility.append(ann_sd)

    #create dictionary to store all the retuns and their volatility from all the scenarios
    data ={'returns':port_returns,'Volatility':port_volatility}
    for counter,ticker in enumerate(portfolio_df.columns.to_list()):
        data[ticker+' weight'] = [w[counter] for w in port_weights]

    #create data frame from dictionnary
    portfolio=pd.DataFrame(data)

    return portfolio

def plot_fig_min(portfolio_1):

    portfolio = get_scenario_portfolio(portfolio_1)

    #minimum volatility:
    min_vol_port=portfolio.iloc[[portfolio['Volatility'].idxmin()]]

    min_weight_df=pd.DataFrame(min_vol_port.iloc[:,2:].unstack()).replace(' weight','').reset_index().rename(columns={'level_0':'ticker',0:'weight'}).drop('level_1',axis=1)
    min_weight_df['ticker']= min_weight_df['ticker'].str.replace(' weight','')

    fig=px.pie(data_frame=min_weight_df,names='ticker',values='weight')
    fig.update_layout(
        title='Weight of minimum volatility portfolio',
        font=dict(size=18 ))
    return fig

def plot_fig_max(portfolio_1):

    portfolio = get_scenario_portfolio(portfolio_1)

    #highest sharpe ratio:
    optimal_sharpe_portfolio=portfolio.loc[[((portfolio['returns']-0)/portfolio['Volatility']).idxmax()]]

    optimal_weight_df=pd.DataFrame(optimal_sharpe_portfolio.iloc[:,2:].unstack()).replace(' weight','').reset_index().rename(columns={'level_0':'ticker',0:'weight'}).drop('level_1',axis=1)
    optimal_weight_df['ticker']= optimal_weight_df['ticker'].str.replace(' weight','')

    fig=px.pie(data_frame=optimal_weight_df,names='ticker',values='weight')
    fig.update_layout(
        title='Weight of portfolio with highest sharpe ratio',
        font=dict(size=18 ))
    return fig

def volume_df(port_row):
    port_row.drop(['volumefrom',], axis=1, inplace=True)
    port_row.rename(columns={'volumeto':'volume'}, inplace=True)
    port_row['year-month'] = port_row['time'].str.slice(0,7,1)
    port_row['year'] = port_row['time'].str.slice(0,4,1)
    return port_row

def volume_df_1(port_row):
    #port_row.drop(['volumefrom',], axis=1, inplace=True)
    port_row.rename(columns={'volumeto':'volume'}, inplace=True)
    port_row['year-month'] = port_row['time'].str.slice(0,7,1)
    port_row['year'] = port_row['time'].str.slice(0,4,1)
    return port_row

def hvplot_volume(test):
    #df.hvplot.line(x=x_var,y=y_var,xlabel =x_label,ylabel =y_label,title=title,groupby=groupby)

    tt = test.hvplot.line(x='time',y='volume',xlabel='Date',ylabel='Volume',title='Intraday Volume',by='ticker',figsize=(500,100),groupby='year')
    return tt

def hvplot_volume_2(test):
    
    #test2.hvplot.line(x='year-month',y='Volume',xlabel='Date',ylabel='Volume',title='Intraday Volume',by='ticker',figsize=(200,100))
    #Trade_data['year']=test['Date'].str.slice(0,4,1)
    tt =test.hvplot.line(x='time',y='Total Trade',xlabel='Date',ylabel='Daily Traded Total',title='Intraday Traded',by='ticker',figsize=(700,1000),groupby='year')
    #s_test= Crypto_data.loc[:,['ETH Close','ETH High']].dropna()
    plt.figure(figsize=(20,15))

    #s_test['ETH Close'].plot(label='close')
    #s_test['ETH High'].plot(label='high')
    #plt.legend(loc='upper right')
    return tt

def plot2(portfolio_1):
    portfolio = get_scenario_portfolio(portfolio_1)
    min_vol_port=portfolio.iloc[[portfolio['Volatility'].idxmin()]]
    optimal_sharpe_portfolio=portfolio.loc[[((portfolio['returns']-0)/portfolio['Volatility']).idxmax()]]
    p1=portfolio.hvplot.scatter(x='Volatility',y='returns',xlabel='Volatility',ylabel='Expected return',legend='top',height=500,width=1000,title='Plot of all portfolios')
    p2=optimal_sharpe_portfolio.hvplot.scatter(x='Volatility',y='returns',marker='o', alpha=.9,)
    p3=min_vol_port.hvplot.scatter(x='Volatility',y='returns')
    #p4=min_vol_port.hvplot.labels(x='Volatility',y='returns',text='minimum variance')
    plot = p1*p2*p3
    return plot


    geo_column = pn.Column(
    "## Population and Crime Geo Plots", monte_carlo_sim_plot, monte_carlo_sim_plot
    )

    scatter_column = pn.Row(
        "## Correlation of Population and Crime Plots",
        sharp_rt_plot(Crypto_data_df),
        corr_plot(Crypto_data_df),
    )

    # Create tabs
    dashboard = pn.Tabs(
        ("Geospatial", geo_column), 
        ("Correlations", scatter_column)
    )