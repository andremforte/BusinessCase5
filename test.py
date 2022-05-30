import streamlit as st
financial_indicators = st.container()
from datetime import date, datetime,timedelta
import yfinance as yf
import hvplot.pandas
import pandas as pd
import numpy as np
import holoviews as hv
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly
import calendar
import plotly.offline as py



# # coin_types = pd.read_html('https://github.com/crypti/cryptocurrencies', header=0)[0]
# df['symbol']=df['Symbol'].str.rstrip('**')
# coin_types['Symbol']=df['symbol']+'-USD'
# st.write(coin_types.head())


hv.extension('bokeh', logo=False)
dataset= st.container()


with financial_indicators:
	st.header('Using indicators to decide when to or when not to trade')
	st.text("Traditional trading indicators are tools used by traders to gauge market sentiment.")



with dataset:
	col1, col2 = st.columns(2)
	
	today = date.today()

	
	# lists=coin_types['symbol'].tolists()
	lists = ['ADA-USD' , 'ATOM-USD', 'AVAX-USD', 'AXS-USD', 'BTC-USD', 'ETH-USD', 'LINK-USD', 'LUNA1-USD', 'MATIC-USD', 'SOL-USD']
# coin_types['Symbol'].to_list()

	df = []


	def download(x):
	    for i in x:
	        data = yf.download(i,
	                           start="2017-04-26",
	                           end=today)
	        df.append(data)


	download(lists)

	[final_ADA, final_ATOM, final_AVAX, final_AXS, final_BTC, final_ETH, final_LINK, final_LUNA1, final_MATIC,
	 final_SOL] = df[0:]

	

	# establishing important lists that may help to create for loops and functions
	dfcurrencies = [final_ADA, final_ATOM, final_AVAX, final_AXS, final_BTC, final_ETH, final_LINK, final_LUNA1,
	                final_MATIC, final_SOL]
	currencies = ['ADA', 'ATOM', 'AVAX', 'AXS', 'BTC', 'ETH', 'LINK', 'LUNA1', 'MATIC', 'SOL']

	categories = st.sidebar.selectbox('Select Coin Type', options = currencies, index = 0)


	# Checking the info in each subdataset
	for i, df in enumerate(dfcurrencies):
	    print("---------------------------------------------------------------------------")
	    print("Information about Dataset of ", currencies[i], ":")
	    print(dfcurrencies[i].info())

	# checking the descriptive statistics of each coin using describe
	for i, value in enumerate(dfcurrencies):
	    print("cryptocurrency: ", currencies[i], "\n", "\n", dfcurrencies[i].describe(include='all'), "\n")


	# Set the short window and long windows
	short_window = 50
	long_window = 100
	# Generate the short and long moving averages (50 and 100 days, respectively)
	for i, df in enumerate(dfcurrencies):
	    df['SMA_7'] = df['Close'].rolling(window=7).mean()
	    df['SMA50'] = df['Close'].rolling(window=short_window).mean()
	    df['SMA100'] = df['Close'].rolling(window=long_window).mean()
	    df['ShortEMA'] = df['Close'].ewm(span=7, adjust=False).mean()

	    df['LongEMA'] = df.Close.ewm(span=26, adjust=False).mean()
	    # Calculate the Moving Average Convergence/Divergence (MACD)
	    df['MACD'] = df['ShortEMA'] - df['LongEMA']
	    # Calcualte the signal line
	    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

	    # Stochastic Oscillator
	    df['14-high'] = df['High'].rolling(14).max()
	    df['14-low'] = df['Low'].rolling(14).min()
	    df['%K'] = (df['Close'] - df['14-low']) * 100 / (df['14-high'] - df['14-low'])
	    df['%D'] = df['%K'].rolling(3).mean()

	    # Other important measures
	    df['spread'] = df['High'] - df['Low']
	    df['volatility'] = df['spread'] / df['Open']
	    df['close_off_high'] = df['Close'] - df['High']
	    df['Signal'] = 0.0

	    # Generate the trading signal 0 or 1,
	    # where 0 is when the SMA50 is under the SMA100, and
	    # where 1 is when the SMA50 is higher (or crosses over) the SMA100
	    df['Signal'][short_window:] = np.where(
	        df['SMA50'][short_window:] > df['SMA100'][short_window:], 1.0, 0.0)
	    # Calculate the points in time at which a position should be taken, 1 or -1
	    df['Entry/Exit'] = df['Signal'].diff()

	    df['Up Move'] = np.nan
	    df['Down Move'] = np.nan
	    df['Average Up'] = np.nan
	    df['Average Down'] = np.nan
	    # Relative Strength
	    df['RS'] = np.nan
	    # Relative Strength Index
	    df['RSI'] = np.nan

		#Calculate Up Move & Down Move
	for i, df in enumerate(dfcurrencies):
		for x in range(1, len(df)):
		    df['Up Move'][x] = 0
		    df['Down Move'][x] = 0
		    
		    if df['Adj Close'][x] > df['Adj Close'][x-1]:
		        df['Up Move'][x] = df['Adj Close'][x] - df['Adj Close'][x-1]
		        
		    if df['Adj Close'][x] < df['Adj Close'][x-1]:
		        df['Down Move'][x] = abs(df['Adj Close'][x] - df['Adj Close'][x-1])  
		        
		## Calculate initial Average Up & Down, RS and RSI
		df['Average Up'][14] = df['Up Move'][1:15].mean()
		df['Average Down'][14] = df['Down Move'][1:15].mean()
		df['RS'][14] = df['Average Up'][14] / df['Average Down'][14]
		df['RSI'][14] = 100 - (100/(1+df['RS'][14]))
		## Calculate rest of Average Up, Average Down, RS, RSI
		for x in range(15, len(df)):
		    df['Average Up'][x] = (df['Average Up'][x-1]*13+df['Up Move'][x])/14
		    df['Average Down'][x] = (df['Average Down'][x-1]*13+df['Down Move'][x])/14
		    df['RS'][x] = df['Average Up'][x] / df['Average Down'][x]
		    df['RSI'][x] = 100 - (100/(1+df['RS'][x]))

		## Calculate the buy & sell signals
		## Initialize the columns that we need
		df['Long Tomorrow'] = np.nan
		df['Buy Signal'] = np.nan
		df['Sell Signal'] = np.nan
		df['Buy RSI'] = np.nan
		df['Sell RSI'] = np.nan
		df['Strategy'] = np.nan
			## Calculate the buy & sell signals
		for x in range(15, len(df)):
		    
		    # Calculate "Long Tomorrow" column
		    if ((df['RSI'][x] <= 40) & (df['RSI'][x-1]>40) ):
		        df['Long Tomorrow'][x] = 1
		    elif ((df['Long Tomorrow'][x-1] == 1) & (df['RSI'][x] <= 70)):
		        df['Long Tomorrow'][x] = 1
		    else:
		    	df['Long Tomorrow'][x] = 0
		        
		    # Calculate "Buy Signal" column
		    if ((df['Long Tomorrow'][x] == 1) & (df['Long Tomorrow'][x-1] == 0)):
		        df['Buy Signal'][x] = df['Adj Close'][x]
		        df['Buy RSI'][x] = df['RSI'][x]
		        
		    # Calculate "Sell Signal" column
		    if ((df['Long Tomorrow'][x] == 0) & (df['Long Tomorrow'][x-1] == 1)):
		        df['Sell Signal'][x] = df['Adj Close'][x]
		        df['Sell RSI'][x] = df['RSI'][x]
		        
		## Calculate strategy performance
		df['Strategy'][15] = df['Adj Close'][15]
		for x in range(16, len(df)):
		    if df['Long Tomorrow'][x-1] == 1:
		        df['Strategy'][x] = df['Strategy'][x-1]* (df['Adj Close'][x] / df['Adj Close'][x-1])
		    else:
		        df['Strategy'][x] = df['Strategy'][x-1]


	def indicators_plot(data, categories, name):
		zip(data, name)
		for i, name in enumerate(name):
			if name==categories:
				df=data[i]
				# plot for SMA in plotly
				fig = go.Figure()
				
				fig.add_trace(go.Scatter(
				    name="Exit",
				    mode="markers", x=df[df['Entry/Exit'] == -1.0].index, y=df[df['Entry/Exit'] == -1.0]['Close'], marker_color='red'
				    
				))
				
				fig.add_trace(go.Scatter(
				    name="Entry",
				    mode="markers", x=df[df['Entry/Exit'] == 1.0].index, y=df[df['Entry/Exit'] == 1.0]['Close'], marker_color='green'
				    
				))
				
				fig.add_trace(go.Scatter(
				    name="Close",
				    mode="lines", x=df.index, y=df["Close"], line_color='grey'
				))

				fig.add_trace(go.Scatter(
				    name="SMA50",
				    mode="lines", x=df.index, y=df["SMA50"], line_color='red'
				))

				fig.add_trace(go.Scatter(
				    name="SMA100",
				    mode="lines", x=df.index, y=df["SMA100"], line_color='gold'
				))

				# Updating layout
				fig.update_layout(
				    title='SMA Indicator',
				    xaxis_title='Date',
				    yaxis_title='Price',
				)

				fig=fig.update_xaxes(showgrid=True, ticklabelmode="period")
				st.plotly_chart(fig)


				# plot for MACD and Signal line
				fig1 = go.Figure()
				fig1.add_trace(go.Scatter(
				    name="MACD",
				    mode="lines", x=df.index, y=df["MACD"],
				))
				fig1.add_trace(go.Scatter(
				    name="Signal_Line",
				    mode="lines", x=df.index, y=df["Signal_Line"],
				    
				))

				
				# Updating layout
				fig1.update_layout(
				    title='MACD indicator',
				    xaxis_title='Date',
				)

				fig1=fig1.update_xaxes(showgrid=True, ticklabelmode="period")
				st.plotly_chart(fig1)



				# plot for RSI
				fig2 = go.Figure()
				
				fig2.add_trace(go.Scatter(
				    name="RSI",
				    mode="lines", x=df.index, y=df["RSI"]
				    
				))
				
				fig2.add_trace(go.Scatter(
				    name="Sell",
				    mode="markers", x=df.index, y=df["Sell RSI"],
				    
				))
				
				fig2.add_trace(go.Scatter(
				    name="Buy",
				    mode="markers", x=df.index, y=df["Buy RSI"],
				))

				# Updating layout
				fig2.update_layout(
				    title='RSI indicator',
				    xaxis_title='Date',
				)

				fig2=fig2.update_xaxes(showgrid=True, ticklabelmode="period")
				st.plotly_chart(fig2)


	indicators_plot(dfcurrencies,categories, currencies)