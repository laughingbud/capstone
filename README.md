# capstone
### Project Title: Advanced Intraday strategies

### **Brief Description**: 
We aim to explore multi-market and trading sessions for intra-day trading opportunities.  More specifically, we want to create a workflow to be able to detect regimes of the market (for example, trending vs. non-trending) and then apply relevant strategies (for example, mean reversion vs. momentum) based on the regime detection results.

### **Installation**: 
!pip install git+https://github.com/laughing-bud/capstone.git

### **Usage**:
The jupyter notebook is currently work in progress. We have done exploratory work which should be visible and bit self explanatory.

### Exploratory researchWe build Bollinger bands with window = 20, band = 2. We show price evolution vis a vis bands and the mean reversion strategy evolution with discrete and continuous signal. This is at lag 0 as we shift the signal by 1 for immediate implementation.
![image](https://github.com/user-attachments/assets/0274a3c6-40c3-4b05-92e5-f33ab9bdca71)

We implement robustness testing on the mean reversion Bollinger band strategy with varying lookback window, bands and lags. Beyond band 2, the strategy detracts as it probably signals breakout and there is strong trend. 
![image](https://github.com/user-attachments/assets/b5350c5a-6908-4180-aa92-dbd5625304bc)

Next, we incorporate the breakout or trend, in the signal by going long when closing price is above band 3 and short when it's below -3. 
![image](https://github.com/user-attachments/assets/b9cb322a-119a-4c06-ad76-d5fa7af923eb)

We plot the bar charts of sharpe ratio for the same set of parameters run on Mean reversion and Mean reversion+trend strategy.

- There is strong alpha decay as we increase the lags, lookback window and the band size.
- Mean reversion + trend, does worse in most cases suggesting a need to build a smarter signal 
![image](https://github.com/user-attachments/assets/2606fa45-cc39-4101-a8ac-b62cbb391771)

--------------------------------------------------------------------------------------------------------------------------------
TREASURY BOND FUTURES WORK (ARCHIVED) - based on yahoo finance hourly data

We note the distinct characteristics in returns, variation in returns (std.dev), higher moments of returns for every hour in the day. There is one hour when the bond futures market is closed. 
![image](https://github.com/user-attachments/assets/3df6f7fc-3a71-4c7a-8270-c136175827ef)
![image](https://github.com/user-attachments/assets/3bc7ddb1-215b-4b94-a353-15c79e2b9b2a)
![image](https://github.com/user-attachments/assets/92e470d0-fda2-4cb0-aa03-3391ece9fa1d)
![image](https://github.com/user-attachments/assets/516f67ea-38ac-49a8-83d3-cca7162d070d)
![image](https://github.com/user-attachments/assets/fab03ff2-12f7-44d4-a8db-7ffbc61914e4)

Overall Hourly returns time series follow a t-distribution (excess kurtosis) which is not very surprising.
![image](https://github.com/user-attachments/assets/ffee704c-2a4a-4cba-bccd-2854ccff6e29)

There is generally a low correlation among hourly returns. However we will dig into this later.
![image](https://github.com/user-attachments/assets/0d71c2fb-2b7a-42d4-b5c6-dce8834b20e1)

The individual hourly returns show particular skewness during certain hours.
![image](https://github.com/user-attachments/assets/bb0c5b68-2bd8-403b-8121-62ada083e619)
![image](https://github.com/user-attachments/assets/cb88a757-8465-48dc-8579-99aff5ab9153)
![image](https://github.com/user-attachments/assets/3f94b280-c539-4aba-a295-fb644ffbb909)
![image](https://github.com/user-attachments/assets/3b98c0ae-e64f-4adb-9b1a-0fa816391d14)
![image](https://github.com/user-attachments/assets/d80cb2d8-f587-4e86-b60b-c2f198ccb937)
![image](https://github.com/user-attachments/assets/dd6c65df-5615-4616-b939-b938b268671c)

The correlation network shows low to moderate correlations among certain hours.
![image](https://github.com/user-attachments/assets/71def29b-0696-4c9c-b5e6-1b66068c1ff5)

The causality network shows more depth of dependency or relations among the hours.
![image](https://github.com/user-attachments/assets/3b3432f4-2a19-40ab-bda3-a98f726a44bd)

![image](https://github.com/user-attachments/assets/76db04f4-5e1e-4b7f-b324-1fe70049413e)


### Strategy highlights 
We created some features such as 'SMA','RSI','MACD','Bollinger_High','Bollinger_Low','Hurst','momentum_rsi', 'trend_macd_diff' and tested whether a Random Forest or XGboost algorithm would be able to predict the next hour returns with a fair accuracy.
![image](https://github.com/user-attachments/assets/cd0c7bfc-7ff7-4d52-9bc7-a3313aeed615)

