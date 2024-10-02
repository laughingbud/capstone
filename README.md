# capstone
### Project Title: Advanced Intraday strategies

### **Brief Description**: 
We aim to explore multi-market and trading sessions for intra-day trading opportunities.  More specifically, we want to create a workflow to be able to detect regimes of the market (for example, trending vs. non-trending) and then apply relevant strategies (for example, mean reversion vs. momentum) based on the regime detection results.

### **Installation**: 
!pip install git+https://github.com/laughing-bud/capstone.git

### **Usage**:
The jupyter notebook is currently work in progress. We have done exploratory work which should be visible and bit self explanatory.

### Exploratory research
You can go through the full research here: https://github.com/laughingbud/capstone/blob/main/Exploratory_research.md

### Preliminary Results
**Mean reversion**
We build Bollinger bands with window = 20, band = 2. We show price evolution vis a vis bands and the mean reversion strategy evolution with discrete and continuous signal. This is at lag 0 as we shift the signal by 1 for immediate implementation.
![image](https://github.com/user-attachments/assets/0274a3c6-40c3-4b05-92e5-f33ab9bdca71)

We implement robustness testing on the mean reversion Bollinger band strategy with varying lookback window, bands and lags. Beyond band 2, the strategy detracts as it probably signals breakout and there is strong trend. 
![image](https://github.com/user-attachments/assets/b5350c5a-6908-4180-aa92-dbd5625304bc)

Next, we incorporate the breakout or trend, in the signal by going long when closing price is above band 3 and short when it's below -3. 
![image](https://github.com/user-attachments/assets/05d50001-5b14-4110-8c34-0e889ac122ed)

We plot the bar charts of sharpe ratio for the same set of parameters run on Mean reversion and Mean reversion+trend strategy.

- There is strong alpha decay as we increase the lags, lookback window and the band size.
- Trend does not add value in most cases suggesting a need to build a smarter signal.
- ![image](https://github.com/user-attachments/assets/e81ea227-688e-4f10-8f29-28d34112e306)

**Seasonality**
We showed in our exploratory research how Bank Nifty returns follow a seasonality pattern of 15 min. We thought to test it for different window sizes.
- Seasonality does look quite significant as our simple betting strategy goes long if there was positive return 15 mins ago and short if it was negative
- We could further enhance the strategy by making the criteria smarter.
![image](https://github.com/user-attachments/assets/eb64822e-98a0-42c4-b8cb-174c7c50cc4a)


--------------------------------------------------------------------------------------------------------------------------------
