import yfinance as yf
import os
import inspect
import rarfile
from git import Repo
from pathlib import Path
import subprocess
import re
import zipfile
from datetime import datetime, timedelta
from dateutil import parser
from pandas.tseries.offsets import BDay
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from scipy.stats import norm, t
import statsmodels.api as sm
import networkx as nx
from statsmodels.tsa.stattools import grangercausalitytests
import ta
from ta import add_all_ta_features
from hurst import compute_Hc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

class Strategy:
    def __init__(self):
        pass

    def split_data(self,data,features,test_size=0.2,position_type='long_only',bid_ask=0,
                   h_period=1):

        bid_ask = 0.03125 if bid_ask == None else bid_ask
        X = data[features].dropna()

        if position_type == 'short_only':
            y = np.where(data['Close'].shift(-h_period) - data['Close'] < bid_ask, 1, 0)  # Binary target
        else:
            if position_type=='long_only':
                y = np.where(data['Close'].shift(-h_period) - data['Close'] > bid_ask, 1, 0)  # Binary target
            elif (position_type=='long_short'):
                y = np.where(data['Close'].shift(-h_period) - data['Close'] < bid_ask,
                                -1, 0)  # Trinary target
                y = np.where(data['Close'].shift(-h_period) - data['Close'] > bid_ask,
                                1, y)  # Trinary target
            else:
                print('Invalid position type')
                return None

        y = y[-X.shape[0]:]
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=test_size,
                                                            shuffle=False)
        # Scale the features
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)

        scaler = StandardScaler().fit(X_test)
        X_test = scaler.transform(X_test)

        y_train_new = np.where(y_train == -1, 2, y_train)
        y_test_new = np.where(y_test == -1, 2, y_test)

        print(f'X_train size: {X_train.shape}')
        print(f'X_test size: {X_test.shape}')
        print(f'y_train size: {y_train.shape}')
        print(f'y_test size: {y_test.shape}')
        return X_train, X_test, y_train, y_test, y_train_new, y_test_new

    def bb_signal(self,data,window: int=20, window_dev: int=2):
        # 1. Calculate Bollinger Bands
        bb_high = ta.volatility.BollingerBands(
            data['close'],window,window_dev).bollinger_hband()
        bb_low = ta.volatility.BollingerBands(
            data['close'],window,window_dev).bollinger_lband()
        # Generate signals
        # bb_sig_discrete = 0
        # bb_sig_cont = 0
        bb_sig_discrete = np.where(
            data['close'] < bb_low, 1,
            np.where(data['close'] > bb_high, -1, 0))
        bb_sig_cont = np.where(data['close'] < bb_low, bb_low-data['close'],
                                     np.where(data['close'] > bb_high,
                                              bb_high-data['close'], 0))
        return bb_high,bb_low,bb_sig_discrete,bb_sig_cont

    def z_score(self,data,window: int=20,z_threshold: int=2):
        ma = data['close'].rolling(window=window).mean()
        std = data['close'].rolling(window=window).std()
        z_score = (data['close'] - ma) / std
        z_signal = np.where(z_score < -z_threshold, 1,
                            np.where(z_score > z_threshold, -1, 0))
        return ma,std,z_signal

    def rsi_signal(self,data,window: int=14):
        rsi = ta.momentum.RSIIndicator(data['close'],window).rsi()
        # Generate signals
        rsi_signal = np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0))
        return rsi,rsi_signal

    def williams_r_signal(self,data,window: int=14):
        williams_r = ta.momentum.WilliamsRIndicator(
            data['High'],data['low'],data['close'],window).williams_r()
        williams_r_signal = np.where(williams_r < -80, 1,
                                   np.where(williams_r > -20, -1, 0))
        return williams_r, williams_r_signal

    def CCI_signal(self,data,window: int=20):
        cci = ta.trend.CCIIndicator(data['High'],data['low'],data['close'],window
                                    ).cci()
        cci_signal = np.where(cci < -100, 1, np.where(cci > 100, -1, 0))
        return cci, cci_signal

    def macd_signal(self,data,window_slow: int=26,window_fast: int=12):
        if window_fast >= window_slow:
            raise ValueError("Fast window should be smaller than slow window")
        macd_indicator = ta.trend.MACD(data['close'],window_slow,window_fast)
        macd = macd_indicator.macd()
        macd_calc = macd_indicator.macd_signal()
        macd_signal = np.where(macd > macd_calc, 1, np.where(macd < macd_calc, -1, 0))
        return macd, macd_calc, macd_signal

    def vwap_signal(self,data,window: int=20):
        vwap = (data['close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
        vwap_signal = np.where(data['close'] > vwap, 1, np.where(data['Close'] < vwap, -1, 0))
        return vwap, vwap_signal

    def ema_signal(self,data,window: int=20):
        ema = data['close'].ewm(span=window, adjust=False).mean()
        ema_signal = np.where(data['close'] > ema, 1, np.where(data['close'] < ema, -1, 0))
        return ema

    def sma_signal(self,data,window: int=20):
        sma = data['close'].rolling(window=window).mean()
        sma_signal = np.where(data['close'] > sma, 1, np.where(data['close'] < sma, -1, 0))
        return sma

    def cmf_signal(self,data,window: int=20):
        cmf = ta.volume.ChaikinMoneyFlowIndicator(
            data['High'],data['low'],data['close'],data['Volume'],window)
        cmf_signal = np.where(cmf > 0, 1, np.where(cmf < 0, -1, 0))
        return cmf, cmf_signal

    def pvt_signal(self,data,window: int=20):
        pvt = (data['close'].pct_change() * data['Volume']).cumsum()
        pvt_signal = np.where(pvt > pvt.shift(1), 1, np.where(pvt < pvt.shift(1), -1, 0))
        return pvt, pvt_signal


    def volume_spike_signal(self,data,window: int=20, threshold: float=1.5):
        # Calculate Volume Spike
        volume_spike = data['Volume'] > data['Volume'].rolling(window=window).mean() * threshold
        volume_spike_signal = np.where(volume_spike, 1, np.where(~volume_spike, -1, 0))
        return volume_spike, volume_spike_signal

    def atr_signal(self,data,window: int=14,ts_hl: int=120,smoothing_hl: int=20,threshold: float=1.0):
        # import talib
        # Calculate ATR
        # atr = talib.ATR(data['High'], data['low'], data['Close'], timeperiod=window)
        atr = ta.volatility.AverageTrueRange(
            data['high'],data['low'],data['close'],window).average_true_range()
        atr_z = ((atr[window-1:] - atr[window-1:].ewm(halflife=smoothing_hl).mean())/atr[window-1:].ewm(halflife=smoothing_hl).std())
        atr_z = atr_z.dropna()

        # Calculate ATR Bands
        ub = data['close'] + atr
        lb = data['close'] - atr
        # Generate signals
        #signal = np.where(data['close'] > ub, -1, np.where(data['close'] < lb, 1, 0))
        # signal = np.where(atr_z < -1*threshold, 1, np.where(atr_z > threshold, -1, 0))
        #signal = pd.DataFrame(signal,atr_z.index,columns=['signal'])
        #signal = pd.concat([signal,data['return'].reindex(signal.index)],axis=1)
        #atr_z = atr_z.ewm(halflife=smoothing_hl).mean()
        #alpha = 0.05*raw_sig.ewm(halflife=20).std()*atr_z
        signal = np.where(atr_z<-1*threshold,1,np.where(atr_z>threshold,-1,0))
        return atr, ub, lb, atr_z, signal

    def donchian_channel_signal(self,data,window: int=20):
        high = data['high'].rolling(window=window).max()
        low = data['low'].rolling(window=window).min()
        donchian_signal = np.where(data['close'] > high, -1, np.where(data['close'] < low, 1, 0))
        return high, low, donchian_signal

    def keltner_channel_signal(self,data,lookback_window: int=20, smoothing_window: int=10):
        ewm = data['close'].ewm(span=smoothing_window).mean()
        atr = ta.volatility.AverageTrueRange(
            data['high'],data['low'],data['close'],lookback_window).average_true_range()
        keltner_high = ewm + 2 * atr
        keltner_low = ewm - 2 * atr
        # keltner_high = data['close'] + 2 * data['close'].rolling(window=window).std()
        # keltner_low = data['close'] - 2 * data['close'].rolling(window=window).std()
        keltner_signal = np.where(data['close'] > keltner_high, -1, np.where(data['close'] < keltner_low, 1, 0))
        return keltner_high, keltner_low, keltner_signal

    def ichimoku_cloud_signal(self,data):
        tenkan_sen = (data['High'].rolling(window=9).max() + data['low'].rolling(window=9).min()) / 2
        kijun_sen = (data['High'].rolling(window=26).max() + data['low'].rolling(window=26).min()) / 2
        senkou_span_a = (tenkan_sen + kijun_sen) / 2
        senkou_span_b = (data['High'].rolling(window=52).max() + data['low'].rolling(window=52).min()) / 2
        # chikou_span = data['close'].shift(-26)
        signal = np.where(data['close'] > senkou_span_a, 1, np.where(data['close'] < senkou_span_a, -1, 0))
        return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, signal

    def mfi_signal(self,data, window: int=14, lb: int=20, ub:int=80):
        # Money flow index reversion
        mfi = ta.volume.MFIIndicator(data['High'],data['low'],data['close'],data['Volume'],window=window).money_flow_index()
        mfi_signal = np.where(mfi < lb, 1, np.where(mfi > ub, -1, 0))
        return mfi, mfi_signal

    def obv_signal(self,data):
        obv = ta.volume.OnBalanceVolumeIndicator(data['close'],data['Volume']).on_balance_volume()
        obv_signal = np.where(obv - obv.shift(1)>0, 1, np.where(obv - obv.shift(1) < 0, -1, 0))
        # obv_signal = np.where(obv > 0, 1, np.where(obv < 0, -1, 0))
        return obv, obv_signal

    def sar_signal(self,data,acceleration: float=0.02,maximum: float=0.2):
        sar = ta.trend.SARIndicator(data['High'],data['low'],
                                    acceleration=acceleration,maximum=maximum
                                    ).sar()
        sar_signal = np.where(data['close'] > sar, 1,
                              np.where(data['close'] < sar, -1, 0))
        return sar, sar_signal

    def force_index_signal(self,data,window: int=20):
        force_index = ta.volume.ForceIndexIndicator(data['close'],data['Volume'],window).force_index()
        force_index_signal = np.where(force_index > 0, 1, np.where(force_index < 0, -1, 0))
        return force_index, force_index_signal

    def stochastic_signal(self,data,k_window: int=14,d_window: int=3):
        stoch = ta.momentum.StochasticOscillator(
            data['High'],data['low'],data['close'],k_window,d_window)
        stoch_signal = np.where(stoch > 80, -1, np.where(stoch < 20, 1, 0))
        return stoch, stoch_signal

    def dmi_signal(self,data,window: int=14):
        dmi = ta.trend.DMIIndicator(data['high'],data['low'],data['close'],window)
        dmi_signal = np.where(dmi > 0, 1, np.where(dmi < 0, -1, 0))
        return dmi, dmi_signal

    def cmo_signal(self,data,window: int=14):
        cmo = ta.trend.CMOIndicator(data['close'],window).cmo()
        cmo_signal = np.where(cmo > 0, 1, np.where(cmo < 0, -1, 0))
        return cmo, cmo_signal

    def adx_signal(self,data,window: int=14):
        adx = ta.trend.ADXIndicator(data['High'],data['low'],data['close'],window).adx()
        adx_signal = np.where(adx > 25, 1, np.where(adx < 25, -1, 0))
        return adx, adx_signal

    def vwma_signal(self,data):
        vwma = (data['close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
        vwma_signal = np.where(data['close'] > vwma, -1, np.where(data['close'] < vwma, 1, 0))
        return vwma, vwma_signal

    def price_spike_signal(self,data,threshold: float=1.05):
        # price_spike = data['close'].diff() > 0
        price_spike = data['close'] > data['close'].rolling(window=20).mean() * threshold
        price_spike_signal = np.where(price_spike, 1, np.where(~price_spike, -1, 0))
        return price_spike, price_spike_signal

    def calculate_analytics(self,returns):
        """Calculates various performance analytics for a given series of returns.

        Args:
          returns: A pandas Series of returns.

        Returns:
          A dictionary containing the calculated analytics.
        """
        most_frequent_time_diff = returns.index.to_series().diff().mode()[0]

        if most_frequent_time_diff == pd.Timedelta(minutes=1) or most_frequent_time_diff == pd.Timedelta(hours=1):
          time_difference = returns[returns.index.date == returns.index.date[0]].index.max() - returns[returns.index.date == returns.index.date[0]].index.min()
          trading_hours = time_difference.total_seconds() / 3600
          trading_hours = 6.25 #override as issues when sample is split intraday
          #print(f'Trading hours:{trading_hours}')
        if most_frequent_time_diff == pd.Timedelta(minutes=1):
          multiplier = 252 * trading_hours * 60  # trading days*trading hours*minutes in an hour
        elif most_frequent_time_diff == pd.Timedelta(hours=1):
          multiplier = 252 * trading_hours  # trading days*trading hours
        elif most_frequent_time_diff == pd.Timedelta(days=1):
          multiplier = 252  # trading days
        else:
          #print ("Unknown Frequency. Assuming 'minutely'.")  # Handle cases where the frequency is not clearly hourly or minutely
          multiplier = 252 * 6.25 * 60  # trading days*trading hours*minutes in an hour
            
        risk_free_rate = 0.02 # annualized cash rate
        #print(f'Risk free rate:{risk_free_rate},Multiplier to annualize input returns:{multiplier}')
        analytics = {}
        analytics['Sharpe Ratio'] = ((returns.mean()*multiplier -risk_free_rate) / (returns.std()*np.sqrt(multiplier))).round(1)
        analytics['Sortino Ratio'] = ((returns.mean()*multiplier -risk_free_rate) / (returns[returns < 0].std()*np.sqrt(multiplier))).round(1)
        analytics['Max Drawdown'] = (returns.cummax() - returns.cumsum()).max()
        analytics['VaR(95%)'] = returns.quantile(0.05)
        analytics['Expected shortfall(95%)'] = returns[returns <= returns.quantile(0.05)].mean()
        # analytics['Annualized Return'] = (1 + returns).prod() ** (252*24 / len(returns)) - 1
        # analytics['Annualized Volatility'] = returns.std() * ((252*24) ** 0.5)
        # analytics['Total return'] = ((1 + returns).cumprod() - 1)[-1]
        analytics['Ann. return'] = (returns.mean()*multiplier).round(3)
        analytics['Ann. vol'] = (returns.std()*np.sqrt(multiplier)).round(3)
        
        return analytics

    def create_mean_reversion_features(self, data):
        # features = pd.DataFrame(index=data.index)

        # 1. Calculate Bollinger Bands
        bb_high,bb_low,bb_signal = self.bb_signal(data)
        # data['BB_High'] = ta.volatility.BollingerBands(data['close']).bollinger_hband()
        # data['BB_Low'] = ta.volatility.BollingerBands(data['close']).bollinger_lband()
        # # Generate signals
        # data['BB_signal'] = np.where(data['close'] < data['BB_low'], 1, np.where(data['close'] > data['BB_High'], -1, 0))

        # 2. Calculate Z-Score
        z_ma,z_std,z_signal = self.z_score(data)
        # data['MA'] = data['Close'].rolling(window=20).mean()
        # data['STD'] = data['Close'].rolling(window=20).std()
        # data['Z-Score'] = (data['Close'] - data['MA']) / data['STD']
        # # Generate signals
        # data['Z_signal'] = np.where(data['Z-Score'] < -2, 1, np.where(data['Z-Score'] > 2, -1, 0))

        # 3. Calculate RSI
        rsi,rsi_signal = self.rsi_signal(data)
        # data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
        # # Generate signals
        # data['RSI_signal'] = np.where(data['RSI'] < 30, 1, np.where(data['RSI'] > 70, -1, 0))

        # 4. Calculate Williams %R
        williams_r, williams_r_signal = self.williams_r_signal(data)
        # data['Williams %R'] = ta.momentum.WilliamsRIndicator(data['High'], data['low'], data['Close']).williams_r()
        # # Generate signals
        # data['W_signal'] = np.where(data['Williams %R'] < -80, 1, np.where(data['Williams %R'] > -20, -1, 0))

        # 5. Calculate CCI
        cci, cci_signal = self.cci_signal(data)
        # data['CCI'] = ta.trend.CCIIndicator(data['High'], data['low'], data['Close']).cci()
        # # Generate signals
        # data['CCI_signal'] = np.where(data['CCI'] < -100, 1, np.where(data['CCI'] > 100, -1, 0))

        # 6. Calculate MACD
        macd, macd_calc, macd_signal = self.macd_signal(data)
        # data['MACD'] = ta.trend.MACD(data['Close']).macd()
        # data['MACD_calc'] = ta.trend.MACD(data['Close']).macd_signal()
        # # Generate signals
        # data['MACD_signal'] = np.where(data['MACD'] > data['MACD_calc'], 1, np.where(data['MACD'] < data['MACD_calc'], -1, 0))

        # 7. Calculate VWAP
        vwap, vwap_signal = self.vwap_signal(data)
        # data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
        # # Generate signals
        # data['VWAP_signal'] = np.where(data['Close'] > data['VWAP'], 1, np.where(data['Close'] < data['VWAP'], -1, 0))

        # 8. Calculate EMA
        data['EMA'] = ta.trend.EMAIndicator(data['Close']).ema_indicator()
        # Generate signals
        data['EWA_signal'] = np.where(data['Close'] > data['EMA'], 1, np.where(data['Close'] < data['EMA'], -1, 0))

        # 9. Calculate SMA
        data['SMA'] = ta.trend.SMAIndicator(data['Close'], window = 10).sma_indicator()
        # Generate signals
        data['SMA_signal'] = np.where(data['Close'] > data['SMA'], 1, np.where(data['Close'] < data['SMA'], -1, 0))

        # 10. Calculate Chaikin Money Flow
        data['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(data['High'], data['low'], data['Close'], data['Volume']).chaikin_money_flow()
        # Generate signals
        data['CMF_signal'] = np.where(data['CMF'] > 0, 1, np.where(data['CMF'] < 0, -1, 0))

        # 11. Calculate PVT
        data['PVT'] = (data['Close'].pct_change() * data['Volume']).cumsum()
        # Generate signals
        data['PVT_signal'] = np.where(data['PVT'] > data['PVT'].shift(1), 1, np.where(data['PVT'] < data['PVT'].shift(1), -1, 0))

        return data

    def create_trend_features(self, data, window=20):
        features = pd.DataFrame(index=data.index)

        # 1. Close Price
        features['Close'] = data['Close']

        # 2. Moving Average Convergence Divergence (MACD)
        features['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
        features['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
        features['MACD'] = features['EMA_12'] - features['EMA_26']
        features['MACD_Signal'] = features['MACD'].ewm(span=9, adjust=False).mean()

        # 3. Relative Strength Index (RSI)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        features['RSI'] = 100 - (100 / (1 + rs))

        # 4. Stochastic Oscillator
        features['Stoch_K'] = ((data['Close'] - data['low'].rolling(window=window).min()) /
                                (data['high'].rolling(window=window).max() - data['low'].rolling(window=window).min())) * 100
        features['Stoch_D'] = features['Stoch_K'].rolling(window=3).mean()  # 3-period smoothing

        # 5. Average True Range (ATR)
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['Close'].shift(1))
        low_close = np.abs(data['low'] - data['Close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features['ATR'] = tr.rolling(window=window).mean()

        # 6. Rate of Change (ROC)
        features['ROC'] = data['Close'].pct_change(periods=window)

        # 7. Momentum
        features['Momentum'] = data['Close'] - data['Close'].shift(window)

        # 8. Cumulative Returns
        features['Cumulative_Returns'] = (1 + data['Close'].pct_change()).cumprod() - 1

        # 9. Volatility
        features['Volatility'] = data['Close'].pct_change().rolling(window=window).std()

        # 10. Bollinger Bands
        features['SMA_Close'] = data['Close'].rolling(window=window).mean()
        features['Upper_Band'] = features['SMA_Close'] + 2 * data['Close'].rolling(window=window).std()
        features['Lower_Band'] = features['SMA_Close'] - 2 * data['Close'].rolling(window=window).std()

        # 11. Volume Weighted Average Price (VWAP)
        features['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()

        # 12. On-Balance Volume (OBV)
        features['OBV'] = (data['Volume'] * np.sign(data['Close'].diff())).cumsum()

        # 13. Accumulation/Distribution Line
        features['AD'] = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low']) * data['Volume']
        features['AD_Line'] = features['AD'].cumsum()

        # 14. Price Action
        features['Price_Action'] = data['Close'].diff()

        # 15. Lagged Price
        features['Lagged_Price'] = data['Close'].shift(1)

        # 16. Exponential Moving Average (EMA)
        features['EMA'] = data['Close'].ewm(span=window, adjust=False).mean()

        # 17. Trend Strength Indicator (Custom)
        features['Trend_Strength'] = features['Close'].diff(window).rolling(window=window).mean()

        return features


    def mean_reversion(self,data,lookback,band_tol):
        if lookback is None:
            # Handle the case where lookback is not provided
            lookback = 4 # Or any other suitable default value
        if band_tol is None:
            band_tol = 2 # Or any other suitable default value
        # Calculate Bollinger Bands
        data['MA20'] = data['Close'].rolling(window=lookback).mean()
        data['STD20'] = data['Close'].rolling(window=lookback).std()
        data['Upper'] = data['MA20'] + (data['STD20'] * band_tol)
        data['Lower'] = data['MA20'] - (data['STD20'] * band_tol)
        data['Buy'] = np.where(data['Close'] < data['Lower'], 1, 0)
        data['Sell'] = np.where(data['Close'] > data['Upper'], -1, 0)
        data['Position'] = data['Buy'] + data['Sell']

        # Shift position to reflect trades
        data['Position'] = data['Position'].shift(1)

        # Calculate returns
        data['Returns'] = data['Close'].pct_change() * data['Position']
        data['Cumulative'] = (1 + data['Returns']).cumprod()
        # display(data)
        # Plot the cumulative returns
        plt.figure(figsize=(12, 6))
        data['Cumulative'].plot()
        plt.title('Strategy Cumulative Returns')
        plt.show()

        initial_capital = 100000
        data['Strategy Value'] = initial_capital * data['Cumulative']

        # Show the final strategy value
        print(f"Final Portfolio Value: {data['Strategy Value'].iloc[-1]:.2f}")
        return None

    def compute_rsi(self, data, window):
        """
        Computes the Relative Strength Index (RSI) for a given dataset.

        :param data: A Pandas Series of prices (usually 'Close' prices).
        :param window: The number of periods to use for RSI calculation (typically 14).
        :return: A Pandas Series representing the RSI.
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi


    def momentum(self,data,st_lb,lt_lb,rsi_lb,rsi_ut,rsi_lt):
        if rsi_lb is None:
            # Handle the case where lookback is not provided
            rsi_lb = 14 # Or any other suitable default value
        if st_lb is None:
            # Handle the case where lookback is not provided
            st_lb = 20 # Or any other suitable default value
        if lt_lb is None:
            # Handle the case where lookback is not provided
            lt_lb = 50 # Or any other suitable default value
        if rsi_ut is None:
            rsi_ut = 70 # Or any other suitable default value
        if rsi_lt is None:
            rsi_lt = 30 # Or any other suitable default value

        # Calculate SMA and RSI
        data['SMA20'] = data['Close'].rolling(window=st_lb).mean()
        data['SMA50'] = data['Close'].rolling(window=lt_lb).mean()
        data['RSI'] = self.compute_rsi(data['Close'], rsi_lb)

        # Define buy/sell signals based on SMA crossover and RSI
        data['Buy'] = np.where((data['SMA20'] > data['SMA50']) & (data['RSI'] < rsi_ut), 1, 0)
        data['Sell'] = np.where((data['SMA20'] < data['SMA50']) & (data['RSI'] > rsi_lt), -1, 0)
        data['Position'] = data['Buy'] + data['Sell']

        # Shift position to reflect trades
        data['Position'] = data['Position'].shift(1)

        # Calculate returns
        data['Returns'] = data['Close'].pct_change() * data['Position']
        data['Cumulative'] = (1 + data['Returns']).cumprod()

        # Plot the cumulative returns
        plt.figure(figsize=(12, 6))
        data['Cumulative'].plot()
        plt.title('Momentum Strategy Cumulative Returns')
        plt.show()

        initial_capital = 100000
        data['Strategy Value'] = initial_capital * data['Cumulative']

        # Show the final strategy value
        print(f"Final Portfolio Value: {data['Strategy Value'].iloc[-1]:.2f}")
        

class trading_strategy:
    def __init__(self,prices):
        self.prices = prices
        self.returns = prices.pct_change(periods=1).dropna(how='all')
        # self.strat_dict = strat_dict
        # print(prices.info())
        pass

    def create_strategy(self,strat_dict):
        self.strat_dict = strat_dict
        lookback = self.strat_dict['lookback']
        rebalance_freq = self.strat_dict['rebalance_freq'] # Get the desired frequency
        strategy_type = self.strat_dict.get('strategy_type', 'long_short') # Added strategy_type with default

        prices = self.prices
        prices = prices.dropna(how='all')
        if self.strat_dict.get('min_xs_count') is not None:
            prices = prices[prices.count(axis=1)>=self.strat_dict.get('min_xs_count')]
            # print(prices.head())
        signal = prices.pct_change(periods=lookback)
        # Reverse the sign on scores to reflect momentum vs mean reversion bets
        if self.strat_dict['strategy'] == 'momentum':
            multiplier = 1
        elif self.strat_dict['strategy'] in ['reversion',
                                             'mean-reversion',
                                             'mean reversion',
                                             'mean_reversion']:
            multiplier = -1
        else:
            multiplier = 1

        strategy_name = self.strat_dict['strategy'] #strategy name
        print(f'Creating a {strategy_name} with lookback of {lookback} days.')
        print(f'Rebalancing frequency: {rebalance_freq}')
        print(f'Strategy Type: {strategy_type}') # Print the strategy type

        signal = signal.dropna(how='all')*multiplier

        xscored = pd.DataFrame() # Initialize xscored

        if self.strat_dict['xscored']:
            xscored = signal.sub(signal.mean(axis=1), axis=0).div(signal.std(axis=1),axis=0).dropna(how='all')

        if 'lags' in self.strat_dict.keys():
            lag_list = self.strat_dict['lags']
        else:
            lag_list = [1]

        store_of_results = {}

        for lag in lag_list:
            # store_of_results['lag'+str(lag)] = {}
            store_of_results[lag] = {}

            # --- Target Weights Calculation (Daily) ---
            # These are the weights we *would* take if we rebalance daily
            target_weights = xscored.shift(lag).dropna(how='all')

            # --- Apply Long-Only or Short-Only filter and scale to sum to 1 ---
            if strategy_type == 'long_only':
                target_weights[target_weights < 0] = 0 # Set negative weights to 0
                # Scale positive weights to sum to 1
                # target_weights = target_weights.div(target_weights.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
            # elif strategy_type == 'short_only':
            #     target_weights[target_weights > 0] = 0 # Set positive weights to 0
            #      # Scale absolute value of negative weights to sum to 1, then multiply by -1
            #     target_weights = target_weights.abs().div(target_weights.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * -1
            elif strategy_type == 'long_short':
                pass # Leave as is, already centered around zero if xscored
            else:
                raise ValueError(f"Invalid strategy_type: {strategy_type}")


            # --- Determine Actual Held Positions based on Rebalancing Frequency ---
            # 1. Align target weights with the returns index (important!)
            target_weights, aligned_returns = target_weights.align(self.returns, join='inner', axis=0)

            # 2. Identify rebalancing timestamps
            # Use resample to get the end-of-period timestamps for the desired frequency
            # Then select only those timestamps from our target_weights index
            rebalance_timestamps = target_weights.resample(rebalance_freq).last().index
            actual_rebalance_dates = target_weights.index.intersection(rebalance_timestamps)

            # 3. Create boolean mask for rebalance days
            # is_rebalance_day = target_weights.index.isin(actual_rebalance_dates)
            is_rebalance_day = target_weights.index.isin(actual_rebalance_dates)


            try:
                held_positions = target_weights.where(pd.DataFrame(
                    np.tile(is_rebalance_day, (target_weights.shape[1], 1)).T,
                    index=target_weights.index,
                    columns=target_weights.columns
                )).ffill()
            except ValueError as e:
                print(f"ValueError occurred at .where(): {e}")
                # Optional: Re-print shapes right before the error if needed
                raise e # Re-raise the error after printing info

            # --- Align held positions with returns again after ffill ---
            held_positions, aligned_returns_final = held_positions.align(
                aligned_returns, join='inner', axis=0)

             # --- Calculate Gross Leverage ---
            gross_leverage_org_strat = held_positions.abs().sum(axis=1)

            # --- Calculate Raw Portfolio Returns (Based on Held Positions) ---
            # Multiply the held weights by the *actual* daily returns
            raw_portfolio_returns = (held_positions * aligned_returns_final).sum(axis=1)
            raw_portfolio_returns = raw_portfolio_returns.dropna() # Drop any remaining NaNs

            # --- Apply Min Cross-Sectional Count Filter (if applicable) ---
            investable_idx = raw_portfolio_returns.index # Start with all days we have returns for


            if self.strat_dict.get('min_xs_count') is not None:
                min_xs_count = self.strat_dict['min_xs_count']

                # Align xscored index before counting
                aligned_xscored = xscored.reindex(raw_portfolio_returns.index)
                # Find dates where the original signal had enough assets
                valid_count_idx = aligned_xscored[aligned_xscored.count(axis=1) >= min_xs_count].index
                # Intersect with the dates we have returns for
                investable_idx = raw_portfolio_returns.index.intersection(valid_count_idx)


            # Filter the raw returns based on investable index
            filtered_portfolio_returns = raw_portfolio_returns.loc[investable_idx]

            if self.strat_dict.get('target_vol') is not None:
                target_vol = self.strat_dict['target_vol']
                annualize_factor = np.sqrt(252)

                sigma_tgt = target_vol / annualize_factor
                lookback = self.strat_dict['lookback']
                # rolling_vol = raw_strategy_returns.rolling(lookback).std().dropna(how='all')
                rolling_vol = filtered_portfolio_returns.rolling(lookback).std().fillna(1.0)

                scaling_factor = (sigma_tgt / rolling_vol).fillna(1.0)
                # Cap scaling factor to avoid extreme leverage (e.g., max 3x)
                scaling_factor = scaling_factor.clip(upper=3.0)
            else:
                scaling_factor = 1

            # Apply scaling factor to the filtered returns
            strategy_returns_final = filtered_portfolio_returns * scaling_factor
            strategy_returns_final = strategy_returns_final.dropna() # Final dropna
            gross_leverage_scaled_strat = held_positions.mul(scaling_factor,axis=0).dropna().sum(axis=1)

            # --- Store Results ---
            if not strategy_returns_final.empty:
                store_of_results[lag] = self.strategy_stats(strategy_returns_final)
                store_of_results[lag]['scaling_factor'] = scaling_factor
                store_of_results[lag]['unscaled_returns'] = filtered_portfolio_returns
                # Store gross leverage, aligned to the strategy_returns_final index
                store_of_results[lag]['gross_leverage_ts'] = gross_leverage_scaled_strat.reindex(strategy_returns_final.index)

                # Store asset returns and scaled holdings for the best lag
                store_of_results[lag]['asset_returns'] = aligned_returns_final.reindex(strategy_returns_final.index)
                store_of_results[lag]['unscaled_holdings'] = held_positions
                store_of_results[lag]['scaled_holdings'] = held_positions.mul(scaling_factor,axis=0).dropna()


            else:
                 print(f"Warning: No valid strategy returns generated for lag {lag} with current settings.")
                 store_of_results[lag] = {} # Store empty dict if no returns\

        # Find max sharpe ratio lead/lag
        max_sharpe = -np.inf
        best_lag = None

        for lag, stats in store_of_results.items():
            if 'sharpe_ratio' in stats and stats['sharpe_ratio'] > max_sharpe:
                max_sharpe = stats['sharpe_ratio']
                best_lag = lag

        print(f"The lead/lag with the maximum Sharpe ratio is: {best_lag}")
        print(f"Maximum Sharpe Ratio: {max_sharpe:.4f}")

        return store_of_results

    def get_best_lag(self,store_of_results):
        # Find max sharpe ratio lead/lag
        max_sharpe = -np.inf
        best_lag = None

        for lag, stats in store_of_results.items():
            if 'sharpe_ratio' in stats and stats['sharpe_ratio'] > max_sharpe:
                max_sharpe = stats['sharpe_ratio']
                best_lag = lag
        return best_lag

    @staticmethod
    def strategy_stats(return_ts: pd.Series, trading_level: int = 100, risk_free_rate: float = 0.0) -> Dict[str, Any]:
        """
        Performs a comprehensive analysis of a trading strategy's return time series.

        This function provides a suite of analytics crucial for a quantitative researcher
        or portfolio manager to evaluate a strategy's performance, risk, and robustness.

        Args:
            return_ts (pd.Series): A pandas Series of daily returns, with a DatetimeIndex.
            trading_level (int, optional): The notional trading level to calculate PnL. Defaults to 100.
            risk_free_rate (float, optional): The annualized risk-free rate for calculations. Defaults to 0.0.

        Returns:
            Dict[str, Any]: A dictionary containing a wide range of performance and risk metrics.
        """
        if not isinstance(return_ts.index, pd.DatetimeIndex):
            raise ValueError("Input 'return_ts' must have a DatetimeIndex.")

        # Ensure return_ts is truly a Series (not a single-column DataFrame)
        if isinstance(return_ts, pd.DataFrame):
            if return_ts.shape[1] == 1:
                return_ts = return_ts.iloc[:, 0] # Convert to Series
            else:
                raise ValueError("Input 'return_ts' must be a Series or a single-column DataFrame.")

        results = {}

        # --- Constants ---
        TRADING_DAYS_PER_YEAR = 252
        daily_risk_free_rate = (1 + risk_free_rate)**(1/TRADING_DAYS_PER_YEAR) - 1

        # --- 1. Top-Line Performance Metrics ---
        total_days = len(return_ts)
        if total_days == 0:
            return {} # Return empty stats if no returns

        # Use geometric mean for more accurate long-term annualized returns
        geo_ann_return = (1 + return_ts).prod() ** (TRADING_DAYS_PER_YEAR / total_days) - 1

        ann_volatility = return_ts.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        # Ensure ann_volatility is a scalar float, handling potential Series or NaN from std()
        if isinstance(ann_volatility, pd.Series):
            if ann_volatility.empty:
                ann_volatility = np.nan
            else:
                ann_volatility = ann_volatility.iloc[0] # Get the scalar value

        # Sharpe Ratio: Measures excess return per unit of total risk
        # Handle zero or NaN volatility explicitly before division
        if pd.isna(ann_volatility) or ann_volatility == 0:
            sharpe_ratio = np.nan
        else:
            sharpe_ratio = (geo_ann_return - risk_free_rate) / ann_volatility

        results['annualized_return'] = geo_ann_return
        results['annualized_volatility'] = ann_volatility
        results['sharpe_ratio'] = sharpe_ratio

        # --- 2. Downside Risk Analytics ---
        # Focus on negative volatility, which is what we actually dislike.
        downside_returns = return_ts[return_ts < daily_risk_free_rate].copy()
        downside_deviation = downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        # Ensure downside_deviation is a scalar float, handling potential Series or NaN
        if isinstance(downside_deviation, pd.Series):
            if downside_deviation.empty:
                downside_deviation = np.nan
            else:
                downside_deviation = downside_deviation.iloc[0]

        # Sortino Ratio: Like Sharpe, but only penalizes for downside volatility.
        if pd.isna(downside_deviation) or downside_deviation == 0:
            sortino_ratio = np.nan
        else:
            sortino_ratio = (geo_ann_return - risk_free_rate) / downside_deviation

        results['downside_deviation'] = downside_deviation
        results['sortino_ratio'] = sortino_ratio

        # --- 3. Drawdown Analysis ---
        # This section is critical. It tells us about the pain the strategy endures.
        equity_curve = (1 + return_ts).cumprod()
        high_water_mark = equity_curve.cummax()
        drawdown_series = (equity_curve / high_water_mark) - 1

        max_drawdown = drawdown_series.min()

        # Calmar Ratio: Return relative to the max drawdown. A measure of risk-adjusted return from a drawdown perspective.
        if max_drawdown == 0 or pd.isna(max_drawdown):
            calmar_ratio = np.nan
        else:
            calmar_ratio = geo_ann_return / abs(max_drawdown)

        # Calculate Drawdown Duration
        drawdown_end_date = drawdown_series.idxmin()
        try:
            drawdown_start_date = high_water_mark.loc[:drawdown_end_date][high_water_mark == high_water_mark.loc[drawdown_end_date]].index[0]
            # Find recovery date: first time equity curve exceeds the HWM at the start of the drawdown
            recovery_mask = equity_curve.loc[drawdown_end_date:] > high_water_mark.loc[drawdown_start_date]
            if recovery_mask.any():
                recovery_date = recovery_mask.idxmax()
                drawdown_duration = (recovery_date - drawdown_start_date).days
            else:
                recovery_date = None # Strategy never recovered
                drawdown_duration = np.nan
        except IndexError:
            drawdown_start_date, recovery_date, drawdown_duration = None, None, 0

        results['max_drawdown'] = max_drawdown
        results['calmar_ratio'] = calmar_ratio
        results['max_drawdown_start'] = drawdown_start_date
        results['max_drawdown_peak'] = drawdown_end_date
        results['max_drawdown_recovery'] = recovery_date
        results['max_drawdown_duration_days'] = drawdown_duration

        # --- 4. Distributional & Win/Loss Statistics ---
        # Understand the nature of the returns themselves.
        win_rate = (return_ts > 0).mean()
        avg_win = return_ts[return_ts > 0].mean()
        avg_loss = return_ts[return_ts < 0].mean()

        # Skew: Is the strategy prone to rare, large losses (negative skew)?
        # Kurtosis: Does the strategy have "fat tails"?
        skewness = return_ts.skew()
        kurtosis = return_ts.kurtosis() # Pandas calculates excess kurtosis (Normal = 0)

        # Tail Ratio: Ratio of 95th percentile gains to 5th percentile losses (95th / abs(5th))
        percentile_95 = return_ts.quantile(0.95)
        percentile_05 = return_ts.quantile(0.05)
        tail_ratio = percentile_95 / abs(percentile_05) if percentile_05 != 0 else np.nan

        results['win_rate'] = win_rate
        results['avg_win_return'] = avg_win
        results['avg_loss_return'] = avg_loss
        results['profit_factor'] = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan
        results['max_drawdown_recovery'] = recovery_date
        results['max_drawdown_duration_days'] = drawdown_duration

        # --- 4. Distributional & Win/Loss Statistics ---
        # Understand the nature of the returns themselves.
        win_rate = (return_ts > 0).mean()
        avg_win = return_ts[return_ts > 0].mean()
        avg_loss = return_ts[return_ts < 0].mean()

        # Skew: Is the strategy prone to rare, large losses (negative skew)?
        # Kurtosis: Does the strategy have "fat tails"?
        skewness = return_ts.skew()
        kurtosis = return_ts.kurtosis() # Pandas calculates excess kurtosis (Normal = 0)

        # Tail Ratio: Ratio of 95th percentile gains to 5th percentile losses (95th / abs(5th))
        percentile_95 = return_ts.quantile(0.95)
        percentile_05 = return_ts.quantile(0.05)
        tail_ratio = percentile_95 / abs(percentile_05) if percentile_05 != 0 else np.nan

        # results['win_rate'] = win_rate
        # results['avg_win_return'] = avg_win
        # results['avg_loss_return'] = avg_loss
        # results['profit_factor'] = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan
        results['skewness'] = skewness
        results['kurtosis'] = kurtosis
        results['tail_ratio_95_05'] = tail_ratio

        # --- 5. PnL and Equity Curves ---
        # Store the core time series for plotting and further analysis.
        # strategy_cmlpnl = (return_ts * trading_level).cumsum()
        # strategy_pnl = return_ts * trading_level
        strategy_pnl = return_ts
        strategy_cmlpnl = (1 + return_ts).cumprod()
        results['strategy_pnl_ts'] = strategy_pnl
        results['strategy_cmlpnl_ts'] = strategy_cmlpnl
        results['equity_curve_ts'] = equity_curve
        results['drawdown_ts'] = drawdown_series
        results['raw_returns_ts'] = return_ts

        return results

    # --- NEW: Visualization Function ---
    @staticmethod
    def plot_strategy_comparison(
        results_dict: Dict[Any, Dict[str, Any]],
        title=None,
        primary_metric: str = 'sharpe_ratio',
        plot_pnl: bool = False,
        rolling_window: int = 126,
        save_chart:bool = False,
    ) -> None:
        """
        Generates a 2x2 dashboard to visually compare strategy backtest results.

        Args:
            results_dict (Dict): A dictionary where keys are strategy parameters (e.g., lead-lags)
                                 and values are the output from the strategy_stats function.
            primary_metric (str): The main metric to compare in the bar chart.
                                  Options: 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'win_rate'.
            plot_pnl (bool): If True, plots cumulative PnL instead of the equity curve in the first subplot.
            rolling_window (int): The window (in days) for the rolling Sharpe ratio calculation. Default is 126 (approx. 6 months).
        """
        if not results_dict:
            print("Results dictionary is empty. Nothing to plot.")
            return

        # --- Setup Plotting Environment ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(3, 2, figsize=(18, 14))
        chart_title = 'Strategy Comparison Dashboard' if title is None else title+' Strategy Comparison Dashboard'
        fig.suptitle(chart_title, fontsize=20, weight='bold')

        # --- 1. Top-Left: Equity Curve / PnL Evolution ---
        ax1 = axes[0, 0]
        plot_type = 'strategy_pnl_ts' if plot_pnl else 'equity_curve_ts'
        y_label = 'Cumulative PnL' if plot_pnl else 'Equity Curve (Starts at 1)'

        for key, data in results_dict.items():
            data[plot_type].plot(ax=ax1, label=key)

        ax1.set_title('Performance Trajectory', fontsize=14, weight='bold')
        ax1.set_ylabel(y_label)
        ax1.legend(title='Lead-Lags')
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

        # --- 2. Top-Right: Primary Metric Comparison ---
        ax2 = axes[0, 1]
        metric_values = {str(k): v[primary_metric] for k, v in results_dict.items() if primary_metric in v}
        metric_series = pd.Series(metric_values)#.sort_values()

        colors = plt.cm.viridis(np.linspace(0.4, 0.95, len(metric_series)))
        bars = ax2.bar(metric_series.index, metric_series.values, color=colors)

        # Add Calmar Ratio as scatter points on a secondary y-axis
        ax2_twin = ax2.twinx()
        calmar_values = {str(k): v['calmar_ratio'] for k, v in results_dict.items() if 'calmar_ratio' in v}
        calmar_series = pd.Series(calmar_values)
        ax2_twin.scatter(calmar_series.index, calmar_series.values, color='red', label='Calmar Ratio', zorder=5)
        ax2_twin.set_ylabel('Calmar Ratio', color='red')
        ax2_twin.tick_params(axis='y', labelcolor='red')
        ax2_twin.legend(loc='upper left')


        # Align the primary and secondary y-axes
        ax2.yaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
        ax2_twin.yaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
        ax2.figure.canvas.draw()  # Draw the canvas to update tick locations

        # Get the tick locations for both axes
        ax2_ticks = ax2.get_yticks()
        ax2_twin_ticks = ax2_twin.get_yticks()

        # Determine the combined minimum and maximum values
        all_values = np.concatenate([metric_series.values, calmar_series.values])
        min_val = np.min(all_values)
        max_val = np.max(all_values)

        # Set the limits for both axes to include all values
        ax2.set_ylim(min_val * 1.1, max_val * 1.1)
        ax2_twin.set_ylim(min_val * 1.1, max_val * 1.1)


        ax2.bar_label(bars, fmt='%.3f', padding=5)
        ax2.set_title(f'{primary_metric.replace("_", " ").title()} Comparison', fontsize=14, weight='bold')
        ax2.set_ylabel(primary_metric.replace("_", " ").title())
        # ax2.set_xlim(left=min(0, metric_series.min() * 1.1))

        # --- 3. Bottom-Left: Drawdown Comparison ---
        ax3 = axes[1, 0]
        drawdown_values = {str(k): v['max_drawdown'] for k, v in results_dict.items() if 'max_drawdown' in v}
        drawdown_series = pd.Series(drawdown_values).sort_values(ascending=False)

        bars = ax3.bar(drawdown_series.index, drawdown_series.values, color='indianred')
        ax3.bar_label(bars, fmt='{:.2%}', padding=3)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))
        ax3.set_title('Maximum Drawdown', fontsize=14, weight='bold')
        ax3.set_ylabel('Peak-to-Trough Loss')

        # --- 4. Bottom-Right: Evolution of Gross Leverage ---
        ax4 = axes[1, 1]
        max_sharpe = -np.inf
        best_lag = None

        for lag, stats in results_dict.items():
            if 'sharpe_ratio' in stats and stats['sharpe_ratio'] > max_sharpe:
                max_sharpe = stats['sharpe_ratio']
                best_lag = lag

        # print(f"The lag with the maximum Sharpe ratio is: {best_lag}")
        # print(f"Maximum Sharpe Ratio: {max_sharpe:.4f}")

        if 'gross_leverage_ts' in results_dict[best_lag]:
              results_dict[best_lag]['gross_leverage_ts'].plot(ax=ax4, label=best_lag)
        else:
              print(f"Warning: 'gross_leverage_ts' not found for strategy {best_lag}. Skipping plot.")

        ax4.set_title('Evolution of Gross Leverage', fontsize=14, weight='bold')
        ax4.set_ylabel('Gross Leverage')
        ax4.legend(title='Lead-Lags')
        ax4.grid(True, which='both', linestyle='--', linewidth=0.5)

        # --- 5. Best sharpe ratio
        ax5 = axes[2,0]
        # Access asset returns for the best lag
        asset_returns = results_dict[best_lag]['asset_returns']

        # Calculate annualized Sharpe ratio for each asset
        annualize_factor = np.sqrt(252)
        asset_sharpe_ratios = (asset_returns.mean() / asset_returns.std()) * annualize_factor

        # Handle potential division by zero
        asset_sharpe_ratios = asset_sharpe_ratios.replace([np.inf, -np.inf], np.nan).dropna()

        asset_sharpe_ratios.sort_values(ascending=False).plot(ax=ax5,kind='bar')
        ax5.set_title(f'Annualized Sharpe Ratio by Asset for Best Lag ({best_lag})')
        ax5.set_xlabel('Asset')
        ax5.set_ylabel('Annualized Sharpe Ratio')
        # plt.xticks(rotation=45, ha='right')


        # --- 6. Best sharpe ratio
        ax6 = axes[2,1]
        # Access scaled holdings for the best lag
        scaled_holdings = results_dict[best_lag]['scaled_holdings']

        # Iterate and plot each asset's scaled holdings
        for column in scaled_holdings.columns:
            scaled_holdings[column].plot(label=column,ax=ax6)

        # Set plot title and labels
        ax6.set_title(f'Scaled Holdings Evolution by Asset for Best Lag ({best_lag})')

        # --- Final Touches ---
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_chart:
            plt.savefig('strategy_dashboard.png')
        # plt.show() # Removed this plt.show() to avoid flickering when other plots are generated

    # ---
    # --- NEW: Walk-Forward Optimization (WFO) Function
    # ---

def run_walk_forward_optimization(
    all_prices: pd.DataFrame,
    all_returns: pd.DataFrame,
    base_strat_dict: dict,
    in_sample_window: pd.DateOffset,
    out_of_sample_window: pd.DateOffset,
    start_date: pd.Timestamp
) -> (pd.Series, Dict[str, Any]):
    """
    Performs a walk-forward optimization on the trading strategy.

    Args:
        all_prices: DataFrame of all historical prices.
        all_returns: DataFrame of all historical returns.
        base_strat_dict: The base strategy configuration dictionary.
        in_sample_window: A DateOffset object for the training period (e.g., pd.DateOffset(years=5)).
        out_of_sample_window: A DateOffset object for the trading period (e.g., pd.DateOffset(months=3)).
        start_date: The timestamp to begin the first in-sample period.

    Returns:
        A tuple containing:
        - The stitched-together out-of-sample returns Series.
        - The final stats dictionary from running strategy_stats on the OOS series.
    """

    oos_returns_list = []
    current_date = start_date
    end_date = all_prices.index.max()

    print("="*40)
    print("Starting Walk-Forward Optimization...")
    print(f"In-Sample Window: {in_sample_window}")
    print(f"Out-of-Sample Window: {out_of_sample_window}")
    print("="*40)

    while current_date + in_sample_window + out_of_sample_window <= end_date:
        # 1. Define window boundaries
        is_start_date = current_date
        is_end_date = current_date + in_sample_window
        oos_start_date = is_end_date + pd.DateOffset(days=1)
        oos_end_date = is_end_date + out_of_sample_window

        print(f"\nProcessing Window:")
        print(f"  In-Sample:   {is_start_date.date()} to {is_end_date.date()}")
        print(f"  Out-of-Sample: {oos_start_date.date()} to {oos_end_date.date()}")

        # 2. Slice data
        is_prices = all_prices.loc[is_start_date:is_end_date]
        # is_returns is not directly used for strategy calculation, only prices are needed
        # is_returns = all_returns.loc[is_start_date:is_end_date]
        oos_prices = all_prices.loc[oos_start_date:oos_end_date]
        # oos_returns = all_returns.loc[oos_start_date:oos_end_date] # This was the problematic line, now replaced by portfolio returns

        if is_prices.empty or oos_prices.empty:
            print("  Skipping window: Not enough data.")
            current_date += out_of_sample_window # Slide to next window
            continue

        # 3. Optimize In-Sample to find best lag
        # Initialize strategy object with In-Sample data
        ts_in_sample = trading_strategy(is_prices)
        # ts_in_sample = ts.create_strategy(base_strat_dict)
        # Run create_strategy to test all lags and get the best one
        store_of_results = ts_in_sample.create_strategy(base_strat_dict)
        best_lag = ts_in_sample.get_best_lag(store_of_results)

        if best_lag is None:
            print("  Skipping window: No best lag found in-sample.")
            current_date += out_of_sample_window
            continue

        print(f"  Found Best In-Sample Lag: {best_lag}")

        # 4. Test Out-of-Sample using ONLY the best_lag
        # Initialize a new strategy object for the OOS test, using the OOS prices
        ts_out_of_sample = trading_strategy(oos_prices)

        # Create a temporary strat_dict for OOS testing, focusing only on the best_lag
        oos_strat_dict = base_strat_dict.copy()
        oos_strat_dict['lags'] = [best_lag] # Only test the best lag
        # Ensure plot is False for OOS to avoid generating plots repeatedly during WFO
        oos_strat_dict['plot'] = False # Assuming 'plot' might be a key in base_strat_dict

        # Run the strategy creation for the OOS period with the best_lag
        oos_results = ts_out_of_sample.create_strategy(oos_strat_dict)

        # Extract the final portfolio returns for the best_lag from the OOS results
        if best_lag in oos_results and 'raw_returns_ts' in oos_results[best_lag]:
            oos_portfolio_returns = oos_results[best_lag]['raw_returns_ts']
            if not oos_portfolio_returns.empty:
                print(f"  OOS Period generated {len(oos_portfolio_returns)} portfolio returns.")
                oos_returns_list.append(oos_portfolio_returns)
            else:
                print("  Skipping window: No valid portfolio returns generated in OOS period.")
        else:
            print(f"  Skipping window: Results for best lag ({best_lag}) not found or no raw_returns_ts in OOS period.")

        # 5. Slide the window
        current_date += out_of_sample_window

    print("\n" + "="*40)
    print("Walk-Forward Optimization Complete.")

    if not oos_returns_list:
        print("Error: No OOS returns were generated. Check data range and window sizes.")
        return pd.Series(dtype=float), {}

    # 6. Stitch all OOS returns together
    final_oos_returns = pd.concat(oos_returns_list)
    final_oos_returns = final_oos_returns.loc[~final_oos_returns.index.duplicated(keep='first')] # Handle overlaps

    print(f"Total OOS returns stitched: {len(final_oos_returns)}")

    # 7. Calculate final stats on the *realistic* OOS equity curve
    final_stats = trading_strategy.strategy_stats(final_oos_returns, 100, base_strat_dict.get('risk_free_rate', 0.0))

    return final_oos_returns, final_stats