from pathlib import Path
from typing import List, NoReturn

import pandas as pd
import numpy as np
import pandas_ta as ta

import yfinance as yf


class Data:
    def __init__(self, raw_data_path: Path) -> NoReturn:
        self.raw_data_path = raw_data_path
    

    def retrieve_tickers(self) -> List[str]:
        df_tickers = pd.read_excel(self.raw_data_path)
        return df_tickers["Tickers"].unique().tolist()


    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Remove duplicates
        df.drop_duplicates(inplace=True)
        
        # Remove zero volume
        df.loc[df['Volume'] <= 0, df.columns] = np.nan
        df.fillna(method='ffill', inplace=True)
        df = df[df['Volume'] > 0]

        # Sort data
        df.sort_index(inplace=True)


    def compute_target(self, df: pd.DataFrame, strategy: str = "binary") -> pd.DataFrame:
        if strategy == "binary":
            pass

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This method calculates multiple technical indicators and their signals,
        using the Adjusted Close price for calculations,
        and appends them as new columns in the provided stock data DataFrame.
        
        Parameters:
        df (pd.DataFrame): DataFrame with columns like 'Open', 'High', 'Low', 'Adj Close', 'Volume'
        
        Returns:
        pd.DataFrame: The input DataFrame with added columns for technical indicators and signals.
        """

        # 1. CCI (Commodity Channel Index)
        df['CCI'] = ta.cci(high=df['High'], low=df['Low'], close=df['Adj Close'], length=20)

        # 2. SAR (Stop and Reverse)
        df[["PSARl", "PSARs", "PSARaf", "PSARr"]] = ta.psar(high=df['High'], low=df['Low'])
        df.loc[df["PSARl"].isna(), "PSARl"] = df.loc[df["PSARs"].notna(), "PSARs"]
        df.rename(columns={"PSARl": "SAR"}, inplace=True)
        df.drop(columns=["PSARs", "PSARaf"], inplace=True)

        # 3. ADX (Average Directional Movement)
        adx = ta.adx(high=df['High'], low=df['Low'], close=df['Adj Close'], length=14)
        df['ADX'] = adx['ADX_14']

        # 4. MFI (Money Flow Index)
        df['MFI'] = ta.mfi(high=df['High'], low=df['Low'], close=df['Adj Close'], volume=df['Volume'], length=14)

        # 5. RSI (Relative Strength Index)
        df['RSI'] = ta.rsi(close=df['Adj Close'], length=14)

        # 6. SK (Slow Stochastic %K)
        stoch = ta.stoch(high=df['High'], low=df['Low'], close=df['Adj Close'], k=14, d=3, smooth_k=3)
        df['SK'] = stoch['STOCHk_14_3_3']

        # 7. SD (Slow Stochastic %D)
        df['SD'] = stoch['STOCHd_14_3_3']

        # 8. RSI-S (RSI Signal using a 9-period moving average of RSI)
        df['RSI-S'] = ta.sma(df['RSI'], length=9)

        # 9. BB-S (Bollinger Bands Signal)
        bb = ta.bbands(close=df['Adj Close'], length=20, std=2)
        df['BB_Upper'] = bb['BBU_20_2.0']
        df['BB_Lower'] = bb['BBL_20_2.0']
        df['BB_Mid'] = bb['BBM_20_2.0']

        # 10. MACD-S (MACD Signal)
        macd = ta.macd(close=df['Adj Close'], fast=12, slow=26, signal=9)
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_Signal'] = macd['MACDs_12_26_9']

        # 11. SAR-S (SAR Indicator Signal)
        df['SAR-S'] = np.sign(df['Adj Close'] - df['SAR'])

        # 12. ADX-S (ADX Indicator Signal - based on trend strength, > 20 implies trend)
        df['ADX-S'] = np.sign(df['ADX'] - 20)

        # 13. S-S (Stochastic Indicator Signal - based on SK and SD crossover)
        df['S-S'] = np.sign(df['SK'] - df['SD'])

        # 14. MFI-S (MFI Indicator Signal - based on crossing the 50 mark)
        df['MFI_previous'] = df['MFI'].shift(1)
        df['MFI-S'] = 0
        df.loc[(df["MFI"] <= 20) & (df["MFI_previous"] > 20), "MFI-S"] = 1
        df.loc[(df["MFI"] >= 80) & (df["MFI_previous"] <= 80), "MFI-S"] = -1
        df.drop(columns="MFI_previous", inplace=True)

        # 15. CCI-S (CCI Indicator Signal)
        df['CCI_previous'] = df['CCI'].shift(1)
        df['CCI-S'] = 0
        df.loc[(df["CCI"] >= 100) & (df["CCI_previous"] < 100), "CCI-S"] = 1
        df.loc[(df["CCI"] <= -100) & (df["CCI_previous"] > -100), "CCI-S"] = -1
        df.drop(columns="CCI_previous")

        # Custom Technical Signals:
        
        # 16. V-S (Sign(Volume - Avg(Volume last 5 days)))
        df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()  # 5-day moving average of volume
        df['V-S'] = np.sign(df['Volume'] - df['Volume_MA5'])

        # 17. CPOP-S (Sign(Adjusted Close Price - Open Price))
        df['CPOP-S'] = np.sign(df['Adj Close'] - df['Open'])

        # 18. CPCPY-S (Sign(Adjusted Close Price - Adjusted Closing Price Yesterday))
        df['Prev_Close'] = df['Adj Close'].shift(1)  # Previous day's Adjusted closing price
        df['CPCPY-S'] = np.sign(df['Adj Close'] - df['Prev_Close'])

        # Drop rows with NaN values (due to rolling/shift operations)
        df.dropna(inplace=True)

        return df


    def fetch_data(self, tickers: List[str]) -> pd.DataFrame:
        pass


    def save(self, output_path: Path) -> NoReturn:
        pass