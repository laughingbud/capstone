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

    def macd_signal(self,data,window_slow: int=26,window_fast: int=1):
        macd = ta.trend.MACD(data['close'],window_slow,window_fast).macd
        macd_calc = ta.trend.MACD(data['close'],window_slow,window_fast).macd_signal()
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

    def atr_signal(self,data,window: int=14):
        # import talib
        # Calculate ATR
        # atr = talib.ATR(data['High'], data['low'], data['Close'], timeperiod=window)
        atr = ta.volatility.AverageTrueRange(
            data['high'],data['low'],data['close'],window).average_true_range()
        # Calculate ATR Bands
        ub = data['close'] + atr
        lb = data['close'] - atr
        # Generate signals
        signal = np.where(data['close'] > ub, -1, np.where(data['close'] < lb, 1, 0))
        return atr, ub, lb, signal

    def donchian_channel_signal(self,data,window: int=20):
        high = data['High'].rolling(window=window).max()
        low = data['low'].rolling(window=window).min()
        donchian_signal = np.where(data['close'] > high, -1, np.where(data['close'] < low, 1, 0))
        return high, low, donchian_signal

    def keltner_channel_signal(self,data,window: int=20):
        ewm = data['close'].ewm(span=window).mean()
        atr = ta.volatility.AverageTrueRange(
            data['High'],data['low'],data['close'],window).average_true_range()
        keltner_high = ewm + 2 * atr
        keltner_low = ewm - 2 * atr
        # keltner_high = data['close'] + 2 * data['Close'].rolling(window=window).std()
        # keltner_low = data['close'] - 2 * data['Close'].rolling(window=window).std()
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

        if most_frequent_time_diff == pd.Timedelta(minutes=1):
          multiplier = 252 * trading_hours * 60  # trading days*trading hours*minutes in an hour
        elif most_frequent_time_diff == pd.Timedelta(hours=1):
          multiplier = 252 * trading_hours  # trading days*trading hours
        elif most_frequent_time_diff == pd.Timedelta(days=1):
          multiplier = 252  # trading days
        else:
          print ("Unknown Frequency")  # Handle cases where the frequency is not clearly hourly or minutely

        risk_free_rate = 0.02 # annualized cash rate

        analytics = {}
        analytics['Sharpe Ratio'] = ((returns.mean()*multiplier -risk_free_rate) / (returns.std()*np.sqrt(multiplier))).round(1)
        analytics['Sortino Ratio'] = (returns.mean()*np.sqrt(multiplier) / returns[returns < 0].std()).round(1)
        analytics['Max Drawdown'] = (returns.cummax() - returns.cumsum()).max()
        analytics['VaR(95%)'] = returns.quantile(0.05)
        analytics['Exp_shortfall(95%)'] = returns[returns <= returns.quantile(0.05)].mean()
        # analytics['Annualized Return'] = (1 + returns).prod() ** (252*24 / len(returns)) - 1
        # analytics['Annualized Volatility'] = returns.std() * ((252*24) ** 0.5)
        # analytics['Total return'] = ((1 + returns).cumprod() - 1)[-1]
        analytics['Ann._return'] = (returns.mean()*multiplier).round(3)
        analytics['Ann._vol'] = (returns.std()*np.sqrt(multiplier)).round(3)
        
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
