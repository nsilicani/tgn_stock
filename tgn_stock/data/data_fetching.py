from pathlib import Path
from typing import Any, Dict, List, NoReturn

from loguru import logger

import pandas as pd
import numpy as np
import pandas_ta as ta

import yfinance as yf

from tqdm import tqdm


class DataRetriever:
    def __init__(self, raw_data_path: Path, config: Dict[str, Any]) -> NoReturn:
        self.raw_data_path = raw_data_path
        self.config = config
        logger.info(f"Load the following cfgs: {self.config}")

    def retrieve_tickers(self) -> List[str]:
        df_tickers = pd.read_excel(self.raw_data_path)
        return df_tickers["Ticker"].unique().tolist()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Remove duplicates
        df.drop_duplicates(inplace=True)

        # Remove zero volume
        df.loc[df["Volume"] <= 0, df.columns] = np.nan
        df.fillna(method="ffill", inplace=True)
        df = df[df["Volume"] > 0]

        # Sort data
        df.sort_index(inplace=True)

        return df

    def compute_target(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config["target_strategy"] == "binary":
            df["diff_1"] = df["Adj Close"].diff(1).shift(-1)
            df["target"] = df["diff_1"].apply(lambda x: 1 if x >= 0 else 0).copy()
            assert df["target"].isna().sum() == 0
            df = df.drop(columns="diff_1")
            return df
        else:
            raise NotImplementedError(
                f"{self.config['strategy']} is not a recognized strategy to compute target"
            )

    def compute_indicators(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        This method calculates multiple technical indicators and their signals,
        using the Adjusted Close price for calculations,
        and appends them as new columns in the provided stock data DataFrame.

        Parameters:
        df (pd.DataFrame): DataFrame with columns like 'Open', 'High', 'Low', 'Adj Close', 'Volume'

        Returns:
        pd.DataFrame: The input DataFrame with added columns for technical indicators and signals.
        """
        logger.info(f"Computing indicators for {ticker}. Total data: {len(df)} ...")

        # 1. CCI (Commodity Channel Index)
        cci_len = self.config["indicators"]["cci"]["len"]
        df["CCI"] = ta.cci(
            high=df["High"], low=df["Low"], close=df["Adj Close"], length=cci_len
        )

        # 2. SAR (Stop and Reverse)
        df[["PSARl", "PSARs", "PSARaf", "PSARr"]] = ta.psar(
            high=df["High"], low=df["Low"]
        )
        df.loc[df["PSARl"].isna(), "PSARl"] = df.loc[df["PSARs"].notna(), "PSARs"]
        df.rename(columns={"PSARl": "SAR"}, inplace=True)
        df.drop(columns=["PSARs", "PSARaf"], inplace=True)

        # 3. ADX (Average Directional Movement)
        adx_len = self.config["indicators"]["adx"]["len"]
        adx_sing_len = self.config["indicators"]["adx"]["sign_len"]
        adx = ta.adx(
            high=df["High"], low=df["Low"], close=df["Adj Close"], length=adx_len
        )
        df["ADX"] = adx[f"ADX_{adx_len}"]
        # ADX-S (ADX Indicator Signal - based on trend strength, > 20 implies trend)
        df["ADX-S"] = np.sign(df["ADX"] - adx_sing_len)

        # 4. MFI (Money Flow Index)
        mfi_len = self.config["indicators"]["mfi"]["len"]
        mfi_low = self.config["indicators"]["mfi"]["mfi_low"]
        mfi_high = self.config["indicators"]["mfi"]["mfi_high"]
        df["MFI"] = ta.mfi(
            high=df["High"],
            low=df["Low"],
            close=df["Adj Close"],
            volume=df["Volume"],
            length=mfi_len,
        )
        # MFI-S (MFI Indicator Signal - based on crossing the 50 mark)
        df["MFI_previous"] = df["MFI"].shift(1)
        df["MFI-S"] = 0
        df.loc[(df["MFI"] <= mfi_low) & (df["MFI_previous"] > mfi_low), "MFI-S"] = 1
        df.loc[(df["MFI"] >= mfi_high) & (df["MFI_previous"] <= mfi_high), "MFI-S"] = -1
        df.drop(columns="MFI_previous", inplace=True)

        # 5. RSI (Relative Strength Index)
        rsi_len = self.config["indicators"]["rsi"]["len"]
        rsi_sign_len = self.config["indicators"]["rsi"]["sign_len"]
        df["RSI"] = ta.rsi(close=df["Adj Close"], length=rsi_len)
        # RSI-S (RSI Signal using a 9-period moving average of RSI)
        df["RSI-S"] = ta.sma(df["RSI"], length=rsi_sign_len)

        # 6. SK (Slow Stochastic %K)
        stoch_k = self.config["indicators"]["stoch"]["k"]
        stoch_d = self.config["indicators"]["stoch"]["d"]
        stoch_smooth_k = self.config["indicators"]["stoch"]["smooth_k"]
        stoch = ta.stoch(
            high=df["High"],
            low=df["Low"],
            close=df["Adj Close"],
            k=stoch_k,
            d=stoch_d,
            smooth_k=stoch_smooth_k,
        )
        if stoch is not None:
            df["SK"] = stoch[f"STOCHk_{stoch_k}_{stoch_d}_{stoch_smooth_k}"]
            df["SD"] = stoch[f"STOCHd_{stoch_k}_{stoch_d}_{stoch_smooth_k}"]
        else:
            war_msg = (
                f"Not possible to compute Stochastic for {ticker} "
                "Setting nan values ..."
            )
            logger.warning(war_msg)
            df["SK"] = np.nan
            df["SD"] = np.nan

        # 8. BB-S (Bollinger Bands Signal)
        bb_len = self.config["indicators"]["bb"]["len"]
        bb_std = self.config["indicators"]["bb"]["std"]
        bb = ta.bbands(close=df["Adj Close"], length=bb_len, std=bb_std)
        df["BB_Upper"] = bb[
            f"BBU_{bb_len}_{bb_std}.0"
        ]  # Note: in pandas_ta dev version the correct column name is: BBU_20_2
        df["BB_Lower"] = bb[f"BBL_{bb_len}_{bb_std}.0"]
        df["BB_Mid"] = bb[f"BBM_{bb_len}_{bb_std}.0"]

        # 9. MACD-S (MACD Signal)
        macd_fast = self.config["indicators"]["macd"]["fast"]
        macd_slow = self.config["indicators"]["macd"]["slow"]
        macd_signal = self.config["indicators"]["macd"]["signal"]
        macd = ta.macd(
            close=df["Adj Close"], fast=macd_fast, slow=macd_slow, signal=macd_signal
        )
        if macd is not None:
            df["MACD"] = macd[f"MACD_{macd_fast}_{macd_slow}_{macd_signal}"]
            df["MACD_Signal"] = macd[f"MACDs_{macd_fast}_{macd_slow}_{macd_signal}"]
        else:
            war_msg = (
                f"Not possible to compute MACD for {ticker} " "Setting nan values ..."
            )
            logger.warning(war_msg)
            df["MACD"] = np.nan
            df["MACD_Signal"] = np.nan

        # 10. SAR-S (SAR Indicator Signal)
        df["SAR-S"] = np.sign(df["Adj Close"] - df["SAR"])

        # 11. S-S (Stochastic Indicator Signal - based on SK and SD crossover)
        df["S-S"] = np.sign(df["SK"] - df["SD"])

        # 12. CCI-S (CCI Indicator Signal)
        df["CCI_previous"] = df["CCI"].shift(1)
        df["CCI-S"] = 0
        cci_low = self.config["indicators"]["cci"]["cci_low"]
        cci_high = self.config["indicators"]["cci"]["cci_high"]
        df.loc[(df["CCI"] >= cci_low) & (df["CCI_previous"] < cci_low), "CCI-S"] = 1
        df.loc[(df["CCI"] <= cci_high) & (df["CCI_previous"] > cci_high), "CCI-S"] = -1
        df.drop(columns="CCI_previous", inplace=True)

        # Custom Technical Signals:

        # 13. V-S (Sign(Volume - Avg(Volume last 5 days)))
        volume_window = self.config["indicators"]["volume"]["window"]
        df["Volume_MA5"] = (
            df["Volume"].rolling(window=volume_window).mean()
        )  # 5-day moving average of volume
        df["V-S"] = np.sign(df["Volume"] - df["Volume_MA5"])

        # 14. CPOP-S (Sign(Adjusted Close Price - Open Price))
        df["CPOP-S"] = np.sign(df["Adj Close"] - df["Open"])

        # 15. CPCPY-S (Sign(Adjusted Close Price - Adjusted Closing Price Yesterday))
        df["Prev_Close"] = df["Adj Close"].shift(
            1
        )  # Previous day's Adjusted closing price
        df["CPCPY-S"] = np.sign(df["Adj Close"] - df["Prev_Close"])
        df.drop(columns="Prev_Close", inplace=True)

        # Drop rows with NaN values (due to rolling/shift operations)
        df.dropna(inplace=True)

        return df

    def fetch_data(self) -> pd.DataFrame:
        # Minimum number of data points required
        min_data_points = self.config["min_data_points"]
        # Initialize an empty list to store data
        data_list = []

        # Loop through each ticker and download historical data
        tickers = self.retrieve_tickers()
        tickers = tickers[: self.config["limit"]] if self.config["limit"] else tickers
        logger.info(f"Fetching data for {len(tickers)} tickers ...")
        for ticker in tqdm(tickers, total=len(tickers)):
            logger.info(f"Ticker: {ticker}")
            df = yf.download(
                ticker, period="max"
            )  # Download all available historical data
            df["Ticker"] = ticker  # Add a column for the ticker symbol
            df = self.clean_data(df)
            df = self.compute_target(df)
            if len(df) > min_data_points:
                df = self.compute_indicators(df, ticker)
                data_list.append(df.copy())
            else:
                logger.warning(
                    f"Required at least {min_data_points} data points for {ticker}. {len(df)} points after cleaning"
                )

        # Combine all data into a single DataFrame
        all_data = pd.concat(data_list)

        logger.success("Data fetching complete.")
        return all_data

    def save(self, df: pd.DataFrame, output_path: Path) -> NoReturn:
        df.to_parquet(output_path)
