# Exploratory Research
## Input data

![image](https://github.com/user-attachments/assets/cab275fd-ffc2-4346-8dd0-122c93266b60)

Index futures tickers: ['BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'NIFTY']
Equity futures tickers: ['AARTIIND', 'ABB', 'ABBOTINDIA', 'ABCAPITAL', 'ABFRL', 'ACC', 'ADANIENT', 'ADANIPORTS', 'ALKEM', 'AMBUJACEM', 'APOLLOHOSP', 'APOLLOTYRE', 'ASHOKLEY', 'ASIANPAINT', 'ASTRAL', 'ATUL', 'AUBANK', 'AUROPHARMA', 'AXISBANK', 
'BAJAJFINSV', 'BAJFINANCE', 'BALKRISIND', 'BALRAMCHIN', 'BANDHANBNK', 'BANKBARODA', 'BATAINDIA', 'BEL', 'BERGEPAINT', 'BHARATFORG', 'BHARTIARTL', 'BHEL', 'BIOCON', 'BOSCHLTD', 'BPCL', 'BRITANNIA', 'BSOFT', 
'CANBK', 'CANFINHOME', 'CHAMBLFERT', 'CHOLAFIN', 'CIPLA', 'COALINDIA', 'COFORGE', 'COLPAL', 'CONCOR', 'COROMANDEL', 'CROMPTON', 'CUB', 'CUMMINSIND', 
'DABUR', 'DALBHARAT', 'DEEPAKNTR', 'DELTACORP', 'DIVISLAB', 'DIXON', 'DLF', 'DRREDDY', 
'EICHERMOT', 'ESCORTS', 'EXIDEIND', 'FEDERALBNK', 'GAIL', 'GLENMARK', 'GMRINFRA', 'GNFC', 'GODREJCP', 'GODREJPROP', 'GRANULES', 'GRASIM', 'GUJGASLTD', 
'HAL', 'HAVELLS', 'HCLTECH', 'HDFC', 'HDFCAMC', 'HDFCBANK', 'HDFCLIFE', 'HEROMOTOCO', 'HINDALCO', 'HINDCOPPER', 'HINDPETRO', 'HINDUNILVR', 
'IBULHSGFIN', 'ICICIBANK', 'ICICIGI', 'ICICIPRULI', 'IDEA', 'IDFC', 'IDFCFIRSTB', 'IEX', 'IGL', 'INDHOTEL', 'INDIACEM', 'INDIAMART', 'INDIGO', 'INDUSINDBK', 'INDUSTOWER', 'INFY', 'INTELLECT', 'IOC', 'IPCALAB', 'IRCTC', 'ITC', 
'JINDALSTEL', 'JKCEMENT', 'JSWSTEEL', 'JUBLFOOD', 'KOTAKBANK', 'LALPATHLAB', 'LAURUSLABS', 'LICHSGFIN', 'LT', 'LTIM', 'LTTS', 'LUPIN', 
'MANAPPURAM', 'MARICO', 'MARUTI', 'MCX', 'METROPOLIS', 'MFSL', 'MGL', 'MOTHERSON', 'MPHASIS', 'MRF', 'MUTHOOTFIN', 
'NATIONALUM', 'NAUKRI', 'NAVINFLUOR', 'NESTLEIND', 'NMDC', 'NTPC', 'OBEROIRLTY', 'OFSS', 'ONGC', 'PAGEIND', 
'PEL', 'PERSISTENT', 'PETRONET', 'PFC', 'PIDILITIND', 'PIIND', 'PNB', 'POLYCAB', 'POWERGRID', 'PVRINOX', 
'RAIN', 'RAMCOCEM', 'RBLBANK', 'RECLTD', 'RELIANCE', 'SAIL', 'SBICARD', 'SBILIFE', 'SBIN', 'SHREECEM', 'SHRIRAMFIN', 'SIEMENS', 'SRF', 'SUNPHARMA', 'SUNTV', 'SYNGENE', 
'TATACHEM', 'TATACOMM', 'TATACONSUM', 'TATAMOTORS', 'TATAPOWER', 'TATASTEEL', 'TCS', 'TECHM', 'TITAN', 'TORNTPHARM', 'TRENT', 'TVSMOTOR', 
'UBL', 'ULTRACEMCO', 'UPL', 'VEDL', 'VOLTAS', 'WIPRO', 'ZEEL', 'ZYDUSLIFE']

We have max 24 months of minutely frequency OHLC, volume and Open interest data. 
![image](https://github.com/user-attachments/assets/2c6538e5-6b53-4ef5-b35a-83656d5b109f)
![image](https://github.com/user-attachments/assets/b024071a-4804-4ef7-a3e5-3e643185f1e7)

Over 70 files for each ticker. Each ticker can have upto three listed futures denoted as F1 (near contract), F2 (next deferred), F3 (deferred contract). As expected volumes and Open interest are much lower for deferred contracts. Hence, much of our focus will be on the near contract F1.
![image](https://github.com/user-attachments/assets/019bc713-7fc5-4e43-ad5f-157dac2b07f3)

![image](https://github.com/user-attachments/assets/c9c7b000-f528-4088-ae55-6c0878a4c5d9)

Below charts focus on Bank Nifty (near) futures contract (F1).
Volume is only autocorrelated for first few minutes and Open interest is only autocorrelated at lag 1.
![image](https://github.com/user-attachments/assets/9bc27e0c-bc4f-4c71-a04f-d9174b2fa992)

Returns are bit nuanced. There is significant seasonality due to autocorrelation being positive at regular intervals like 15,30,45 and 60.
There is also strong autocorrelation at lag 1 but is followed by a strong reversal at lag 2.
![image](https://github.com/user-attachments/assets/c321a8f0-00d6-4f7d-9f7a-ed3bf930d671)

Correlation between return, volume and open interest and even changes in volume and OI don't look significant..
![image](https://github.com/user-attachments/assets/b5ce0fc0-e77d-4faf-8262-cf4f5ebf7618)
![image](https://github.com/user-attachments/assets/557b208a-f151-4aa8-9171-282b3d4dc5c8)

We build Bollinger bands with window = 20, band = 2. We show price evolution vis a vis bands and the mean reversion strategy evolution with discrete and continuous signal. This is at lag 0 as we shift the signal by 1 for immediate implementation.
![image](https://github.com/user-attachments/assets/ea949b16-760e-4b1b-802e-ac9eafeb1ce9)

We implement robustness testing on the mean reversion Bollinger band strategy with varying lookback window, bands and lags. Beyond band 2, the strategy detracts as it probably signals breakout and there is strong trend.
![image](https://github.com/user-attachments/assets/ec0451df-b420-415e-834e-3ce83bbc0395)

Next, we incorporate the breakout or trend, in the signal by going long when closing price is above band 3 and short when it's below -3.
![image](https://github.com/user-attachments/assets/2cb820ef-0d1d-43be-92aa-b7d31024c282)

We plot the bar charts of sharpe ratio for the same set of parameters run on Mean reversion and Mean reversion+trend strategy.
- There is strong alpha decay as we increase the lags, lookback window and the band size.
- Mean reversion + trend, does worse in most cases suggesting a need to build a smarter signal
![image](https://github.com/user-attachments/assets/2bc023f7-2bdd-4a8a-9928-7473dc31fb7b)
