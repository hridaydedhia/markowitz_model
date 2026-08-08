import numpy as np
import pandas as pd
import yfinance as yf 
import matplotlib.pyplot as plt
import scipy.optimize as optimization

NUM_TRADING_DAYS=252
NUM_PORTFOLIOS=10000
#Stocks
#stocks=['AAPL', 'WMT', 'TSLA','GE','AMZN','DB']
stocks=['COLPAL.NS','ICICIBANK.NS','LICHSGFIN.NS','LT.NS','IRFC.NS','RELIANCE.NS','TCS.NS','HDFCBANK.NS','HINDUNILVR.NS','INFY.NS','ITC.NS','KOTAKBANK.NS','MARUTI.NS','NESTLEIND.NS','SBIN.NS','SUNPHARMA.NS','TITAN.NS','ULTRACEMCO.NS']

#Historical data
start_date='2012-01-01'
end_date='2026-01-01'

def download_data():
    #name of stock as keys and stock values as values
    stock_data={}

    for stock in stocks:
      ticker=yf.Ticker(stock)
      stock_data[stock]=ticker.history(start=start_date, end=end_date)['Close']

    return pd.DataFrame(stock_data)

def show_data(data):
   data.plot(figsize=(10,5))
   plt.show()

def calculate_returns(data):
   log_return=np.log(data/data.shift(1))
   return log_return[1:]

def show_statics(returns):
   print(returns.mean()* NUM_TRADING_DAYS)
   print(returns.cov()*NUM_TRADING_DAYS)


def show_mean_variance(returns,weight):
   portfolio_return=np.sum(returns.mean()*weight)*NUM_TRADING_DAYS
   portfolio_variance=np.sqrt(np.dot(weight.T,np.dot(returns.cov()* NUM_TRADING_DAYS,weight)))
   print(f"Expected Portfolio Return: {portfolio_return}")
   print(f"Expected Portfolio Volatility (Standard Deviation): {portfolio_variance}")

def generate_portfolios(returns):
   portfolio_mean=[]
   portfolio_risk=[]
   portfolio_weights=[]

   for _ in range(NUM_PORTFOLIOS):
      w=np.random.random(len(stocks))
      w/=np.sum(w)
      portfolio_weights.append(w)
      portfolio_mean.append(np.sum(returns.mean()*w)*NUM_TRADING_DAYS)
      portfolio_risk.append(np.sqrt(np.dot(w.T,np.dot(returns.cov()*NUM_TRADING_DAYS,w))))
   return np.array(portfolio_weights), np.array(portfolio_mean), np.array(portfolio_risk)

def show_portfoilios(returns,volatilities):
   plt.figure(figsize=(10,6))
   plt.scatter(volatilities,returns,c=returns/volatilities,marker='o')
   plt.grid(True)
   plt.xlabel('Expected Volatility')
   plt.ylabel('Expected Return')
   plt.colorbar(label='Sharpe Ratio')
   plt.show()

def statistics(weights,returns):
   portfolio_return=np.sum(returns.mean()*weights)*NUM_TRADING_DAYS
   portfolio_volatility=np.sqrt(np.dot(weights.T,np.dot(returns.cov()* NUM_TRADING_DAYS,weights)))
   return np.array([portfolio_return,portfolio_volatility,portfolio_return/portfolio_volatility])

#scipi optimization module can find the minimum of a given fucntion, so if we want to maximize then we have add a negatuve sign
def min_function_sharpe(weights,returns):
   return -statistics(weights,returns)[2]

def optimize_portfolio(weights,returns):
   constraints={'type':'eq','fun':lambda x:np.sum(x)-1}
   bounds=tuple((0,1) for x in range(len(stocks)))
   return optimization.minimize(fun=min_function_sharpe,x0=weights[0],args=returns,method='SLSQP',bounds=bounds,constraints=constraints)

def print_optimal_portfolio(optimum, returns):
    print("Optimal portfolio: ", optimum['x'].round(3))
    print("Expected return, volatility and Sharpe ratio: ",statistics(optimum['x'].round(3), returns))

def show_optimal_portfoilios(opt,rets,portfolio_rets,portfolio_vols):
   plt.figure(figsize=(10,6))
   plt.scatter(portfolio_vols,portfolio_rets,c=portfolio_rets/portfolio_vols,marker='o')
   plt.grid(True)
   plt.xlabel('Expected Volatility')
   plt.ylabel('Expected Return')
   plt.colorbar(label='Sharpe Ratio')
   plt.plot(statistics(opt['x'],rets)[1],statistics(opt['x'],rets)[0],'g*',markersize=15.0)
   plt.show()

if __name__=="__main__":
   
   dataset=download_data()
   show_data(dataset)
   calculate_returns(dataset)
   log_daily_returns=calculate_returns(dataset)
   #show_statics(log_daily_returns)
   pweights, means, risk=generate_portfolios(log_daily_returns)

   show_portfoilios(means,risk)
   optimum=optimize_portfolio(pweights,log_daily_returns)
   print_optimal_portfolio(optimum,log_daily_returns)
   show_optimal_portfoilios(optimum,log_daily_returns,means,risk)

   