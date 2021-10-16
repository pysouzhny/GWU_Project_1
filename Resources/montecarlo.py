import os
import numpy as np
import pandas as pd

#import matplotlib.pyplot as plt
from numpy import random

random.seed(32)


def monte_carlo_sim(df=None, trials=1000, sim_days= 252 ,weights=None):
    """
    Returns a data frame with monte-carlo simulation results.
    df must be provided in the same format it is obtained
    trhough IEX-Cloud get_historical_data() function.
    The resulted dataframe will come out with the compound
    product of the daily returns of the portfolio after applying
    the weights provided. If no weights are provided, the function
    will imply that they all have the same weight.
    
    Parameters:
    df: dataframe with the daily close prices of the tickers.
    Ideally, the df must come from IEX Cloud API. If not, make sure
    it's columns are 2-level indexed with the upper index containing
    the ticker and with one column named "close" in the lower level.
    trials: number of trial for the monte-carlo simulation. By 
    default, it is 1000.
    sim_days: amount of simulated days desired. By default 252 (1 year)
    weights: the weights for each portfolio in the same order that the
    prices datframe has the tickers. If no weights list is provided,
    the funtion will imply they all have the same weight
    """
    ##So far, can handle any kind of dataframe
    ##as long as it has 2 levels, where the first level is the ticker and the second
    ##has the HOLC values and one of the columns has the name "close".
    print(f"####Montecarlo####\nDF Columns: {df.columns}\nTrials: {trials}, Simulation days: {sim_days}\n Weights: {weights}")
    #Dataframe formatting:
    if(df.columns.nlevels>1):
        df = normalize_dataframe(df)
    daily_rteurns = df.pct_change().copy()
    
    #checks if the weights provided are correct:
    
    #first it checks if the amount of weights provided matches 
    #the amount of columns of the dataframe to prevent the function
    #to crash. If the weights are not provided, then
    #it will assign an equivalent value to all the weights.
    #If weights are provided but they don't match the amount of
    #columns, then it'll return a explanatory message.
    if weights is None:
        weights = []
        for colm in range(daily_rteurns.shape[1]):
                weights.append(1/daily_rteurns.shape[1]) 
    elif len(weights) is not daily_rteurns.shape[1]:
            return f"weights ({len(weights)}) list must have the same amount of tickers ({daily_rteurns.shape[1]})"
           
    
    #Then it checks if the weights' total is 1.0 to prevent the user to set
    #funny weights. If it's not, returns explanatory message.
    else:
        if round(sum(weights),4) != 1.0:
            
            return f"weights must total 1.0, not {sum(weights)}"
    
    #Calculate avarages and standard deviations for each ticker
    means = daily_rteurns.mean()
    stds = daily_rteurns.std()
    variances = daily_rteurns.var()
    
    #sets current prices as the latest price from dataframe of prices
    current_pirces = pd.DataFrame(df.iloc[-1])
    #Transposes the dataframe to facilitate future handling of the df
    current_pirces = current_pirces.transpose()

    #creats an empty dataframe to store the monte-carlo simulation results
    monte_carlo_cum_ret = pd.DataFrame()
    
    # Seed the monte carlo simulation
    
    
    #Repeats the simulation for "trials" times
    for n in range(trials):
        progress_pct = n/trials
        print('\r',f"Monte-Carlo simulation progress: " , '[{:>7.2%}]'.format(progress_pct), end='')
        #print("")

        #sets prices as current prices
        prices = current_pirces


        #Repeats simulation for "sim_days" times
        for i in range(sim_days):

            #creates an empty dict to store simulated prices
            #of every ticker of a single day
            simulated_prices = {}


            #loops through every ticker last price availabe
            for stock,price in prices.iloc[-1].items():

                #sets a new random price assuming a normal distribution with the 
                #avarage and standard deviation obtained from prices dataframe 
                #new_price = price * (1 + np.random.normal(means[stock],stds[stock])) #old monte -carlo
                daily_drift = means[stock] - (variances[stock]/2)
                drift = daily_drift - 0.5 * stds[stock] ** 2
                diffusion = stds[stock] * np.random.laplace()
                new_price = price * np.exp(drift + diffusion)

                #Adds the key, value pair of the ticker with the new price simulated
                #to the simulated_prices dict.
                simulated_prices.update({stock : [new_price]})


            #Converts the dict to a dataframe to stack it up and build up 
            #the monte-carlo dataframe of a single simulation of "sim_days" days
            new_sim_prices_df = pd.DataFrame.from_dict(simulated_prices)
            prices = pd.concat([prices,new_sim_prices_df],axis = 0,ignore_index=True)
            
            

        #converts the dataframe of simulated prices into a daily return df
        simulated_daily_returns = prices.pct_change()

        #converts the portfolio of daily returns of different stocks into a
        #single-column dataframe of the total returns per day after applying weihgts
        portfolio_daily_returns = simulated_daily_returns.dot(weights)

        #stacks horizontally the single simulation result to build the final monte-carlo
        #dataframe after the "trials" trials.
        monte_carlo_cum_ret[n] = (1 + portfolio_daily_returns.fillna(0)).cumprod()
        

    monte_carlo_cum_ret.head()

    return monte_carlo_cum_ret
    
