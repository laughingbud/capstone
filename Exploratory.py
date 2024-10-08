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

class Exploratory():
    def __init__(self):
        pass

    def plot_corr(self,df,title):
        # Compute the correlation matrix
        corr = df.corr()
        # corr = corr.sort_values(by=[yf_data.columns[0]], ascending=False)
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        # sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True,
        #             linewidths=.5, cbar_kws={"shrink": .5},
        #             annot=True, fmt=".2f")
        sns.heatmap(corr, cmap="YlGnBu",mask=mask)
        plt.title(title)


    def plot_autocorrelations(self,df,maxlags=30,label=''):
        fig,ax = plt.subplots(2,df.shape[1],figsize=(18,5))
        for n in range(df.shape[1]):
            # model = self.best_arima(df.iloc[:,n])
            y_lim = min(1,(1.5*df.iloc[:,n].autocorr(lag=1)).round(2))
            # print(y_lim)
            sm.graphics.tsa.plot_acf(df.iloc[:,n], lags=range(1,maxlags),
                                     ax=ax[0,n])
            ax[0,n].set_title(f'ACF: {df.columns[n]} {label}')
            ax[0,n].set_ylim([-y_lim, y_lim])
            sm.graphics.tsa.plot_pacf(df.iloc[:,n], lags=range(1,maxlags),
                                      ax=ax[1,n])
            ax[1,n].set_title(f'PACF: {df.columns[n]} {label}')
            ax[1,n].set_ylim([-y_lim, y_lim])
        plt.tight_layout()

    def plot_histograms(self,df,label='returns'):

        rows = 1
        cols = df.shape[1]
        fig,ax = plt.subplots(rows,cols,figsize=(18,3))

        for n in range(cols):
            mean = np.mean(df[df.columns[n]])
            stdev = np.std(df[df.columns[n]])
            skewness = skew(df[df.columns[n]])
            kurt = kurtosis(df[df.columns[n]])
            nu, mu, sigma = t.fit(df[df.columns[n]])

            x=np.linspace(mean-4*stdev, mean + 4*stdev, 100)
            ax[n].plot(x,norm.pdf(x, mean, stdev), "r", label="normal")
            ax[n].plot(x,t.pdf(x, nu,mu, sigma), "g", label="t-dist")
            # df[df.columns[n]].plot(kind='hist', bins=50,
            #                                   title=df.columns[n],ax=ax[n])
            ax[n].legend(loc="upper right")
            ax[n].set_title(df.columns[n])
            df[df.columns[n]].hist(bins=50,density=True,histtype='stepfilled',
                                   alpha=0.5,ax=ax[n])
            ax[n].set_xlabel(label)
            ax[n].set_ylabel('frequency')
            ax[n].text(0.02, 0.9, f'Mean: {mean:.3f}', fontsize=9,transform=ax[n].transAxes)
            ax[n].text(0.02, 0.8, f'Std dev: {stdev:.2f}', fontsize=9,transform=ax[n].transAxes)
            ax[n].text(0.02, 0.7, f'Skewness: {skewness:.1f}', fontsize=9,transform=ax[n].transAxes)
            ax[n].text(0.02, 0.6, f'Kurtosis: {kurt:.1f}', fontsize=9,transform=ax[n].transAxes)

        plt.tight_layout()

    def correlation_network(self,corr_matrix,threshold=0.15):
        # corr_matrix = correlation_matrix.copy()
        # Create the correlation network
        G = nx.Graph()

        # Add edges based on correlation threshold
        threshold = threshold  # Adjust this threshold as needed
        for i in corr_matrix.columns:
            for j in corr_matrix.columns:
                if i != j and abs(corr_matrix.loc[i, j]) > threshold:
                    G.add_edge(i, j, weight=corr_matrix.loc[i, j])

        # Draw the network
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=1500, font_size=10)
        plt.title(f'Correlation Network (threshold={threshold})')
        plt.show()

    def causal_network(self,df,maxlag=1):
        G = nx.Graph()
        maxlag = maxlag
        # Example: Test causality between all pairs
        for i in df.columns:
            for j in df.columns:
                if i != j:
                    result = grangercausalitytests(df[[i, j]], maxlag=maxlag, verbose=False)
                    p_values = [round(result[lag][0]['ssr_ftest'][1], 4) for lag in range(1, maxlag+1)]
                    if min(p_values) < 0.05:
                        G.add_edge(i, j, weight=min(p_values))  # Directed edge if i Granger-causes j

        # Draw the causal network
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightgreen', edge_color='gray',
                node_size=1500, font_size=10, arrows=True)
        plt.title(f'Causal Network (maxlag={maxlag})')
        plt.show()

    def granger_causality(self,df,maxlag=1):
        maxlag = maxlag  # Set the maximum lag to consider
        test_results = {}

        # Perform Granger causality tests for each pair of hourly returns
        for hr1 in df.columns:
          for hr2 in df.columns:
            if hr1 != hr2:
              # Test if hr1 causes hr2
              result = grangercausalitytests(df[[hr2, hr1]], maxlag=maxlag,
                                            verbose=False)
              p_values = [round(result[i+1][0]['ssr_ftest'][1], 4) for i in range(maxlag)]
              test_results[(hr1, hr2)] = p_values

        # Print the results
        print("Granger Causality Test Results:")
        for (hr1, hr2), p_values in test_results.items():
          # if showcausal & (p_value < 0.05):
          print(f"Does {hr1} cause {hr2}?")
          for i, p_value in enumerate(p_values):
            print(f"Lag {i+1}: p-value = {p_value}")
            if p_value < 0.05:
              print(f"  => Significant evidence of causality at lag {i+1}")
          print("----")

    def causal_matrix(self, df, maxlag=1):
        """
        Creates a matrix highlighting causal relationships based on Granger causality tests.

        Args:
            df (pd.DataFrame): DataFrame containing time series data.
            maxlag (int): Maximum lag to consider for Granger causality tests.

        Returns:
            pd.DataFrame: A DataFrame representing the causal matrix, where 1 indicates causality
                          and 0 indicates no causality.
        """

        causal_matrix = pd.DataFrame(index=df.columns, columns=df.columns, dtype=int)
        causal_matrix[:] = 0  # Initialize with no causality

        for hr1 in df.columns:
            for hr2 in df.columns:
                if hr1 != hr2:
                    result = grangercausalitytests(df[[hr2, hr1]], maxlag=maxlag, verbose=False)
                    p_values = [round(result[i + 1][0]['ssr_ftest'][1], 4) for i in range(maxlag)]
                    if min(p_values) < 0.05:
                        causal_matrix.loc[hr1, hr2] = 1  # Mark causality if p-value < 0.05

        return causal_matrix


    def causal_strength_matrix(self, df, maxlag=1):
        """
        Creates a matrix capturing the strength of causal relationships
        based on Granger causality tests.

        Args:
            df (pd.DataFrame): DataFrame containing time series data.
            maxlag (int): Maximum lag to consider for Granger causality tests.

        Returns:
            pd.DataFrame: A DataFrame representing the causal strength matrix,
                          where values indicate the minimum p-value found for
                          significant causal relationships. Non-significant
                          relationships are represented as NaN.
        """

        causal_matrix = pd.DataFrame(index=df.columns, columns=df.columns, dtype=float)
        causal_matrix[:] = np.nan  # Initialize with NaN for non-significant relationships

        for hr1 in df.columns:
            for hr2 in df.columns:
                if hr1 != hr2:
                    result = grangercausalitytests(df[[hr2, hr1]], maxlag=maxlag, verbose=False)
                    p_values = [round(result[i + 1][0]['ssr_ftest'][1], 4) for i in range(maxlag)]
                    min_p_value = min(p_values)
                    if min_p_value < 0.05:
                        causal_matrix.loc[hr1, hr2] = min_p_value  # Store the minimum p-value

        # Visualize the causal strength matrix as a heatmap (showing only causal relationships)
        plt.figure(figsize=(10, 8))
        sns.heatmap(causal_matrix.round(2), annot=True, cmap="YlGnBu", cbar=True, mask=causal_matrix.isnull())
        plt.title("Causal Strength Matrix Heatmap")
        plt.xlabel("Effect (Caused)")
        plt.ylabel("Cause (Causing)")
        plt.show()

        return causal_matrix

    def tscore(self,df,halflife,demean=True):
        df.sort_index(axis=0,inplace=True)
        dfmean = df.ewm(halflife=halflife).mean()
        dfstd = df.ewm(halflife=halflife).std()
        if demean:
            return (df-dfmean)/dfstd,dfmean,dfstd
        else:
            return df/dfstd,dfmean,dfstd

    def xscore(self,df):
        ndims = df.ndim
        if ndims == 2:
            return df.sub(df.mean(axis=1),axis=0).div(df.std(axis=1),axis=0)
        elif ndims==3:
            return df.sub(df.mean(axis=1,level=1),axis=0).div(df.std(axis=1,level=1),
                                                              axis=0)
        else:
            print('Error! The number of dimensions should be 2 or max 3.')

#exp = Exploratory()
