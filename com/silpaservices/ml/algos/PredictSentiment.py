import pandas as pd
from sklearn import preprocessing, svm
import sklearn.cross_validation
from sklearn.linear_model import LinearRegression
import numpy
import quandl as q
import math as math


class StockData:

    forecast_out = ''
    stock_symbol = ''

    def __init__(self, stock_symbol):
        self.stock_symbol = stock_symbol

    @staticmethod
    def out_forecast(df):
        return int(math.ceil(0.01 * len(df)))

    def get_stock_data(self, stock_symbol):
        df = q.get('WIKI/' + stock_symbol)
        df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
        df['HL_PCT'] = ((df['Adj. High'] - df['Adj. Close']) / df['Adj. High']) * 100.0
        df['PCT_change'] = ((df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']) * 100.0
        df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
        forecast_col = 'Adj. Close'
        df.fillna(-99999, inplace=True)
        self.forecast_out = self.out_forecast(df)
        df['label'] = df[forecast_col].shift(-self.forecast_out)
        df.dropna(inplace=True)
        return df

    def test(self):
        print(self.get_stock_data(self.stock_symbol))
