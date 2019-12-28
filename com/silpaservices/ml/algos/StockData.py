
import quandl as q
import math as math
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')


class StockData():

    forecast_out = ''
    ticker = ''
    df = ''
    X = ''
    X_lately = ''
    y = ''
    clf = ''
    def __init__(self, ticker):
        self.ticker = ticker

    @staticmethod
    def out_forecast(df):
        return int(math.ceil(0.01 * len(df)))

    def get_stock_data(self):
        self.df = q.get('WIKI/' + self.ticker)
        self.df = self.df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
        self.df['HL_PCT'] = ((self.df['Adj. High'] - self.df['Adj. Close']) / self.df['Adj. High']) * 100.0
        self.df['PCT_change'] = ((self.df['Adj. Close'] - self.df['Adj. Open']) / self.df['Adj. Open']) * 100.0
        self.df = self.df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
        forecast_col = 'Adj. Close'
        self.df.fillna(-99999, inplace=True)
        self.forecast_out = self.out_forecast(self.df)
        self.df['label'] = self.df[forecast_col].shift(-self.forecast_out)
        #self.df.dropna(inplace=True)
        return self.df

    def stock_test(self):
        self.X = np.array(self.df.drop(['label'], 1))

        self.X = preprocessing.scale(self.X)

        self.X_lately = self.X[-self.forecast_out:]
        self.X = self.X[:-self.forecast_out:]


        self.df.dropna(inplace=True)

        self.y = np.array(self.df['label'])

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, train_size=0.2)

        self.clf = LinearRegression()
        self.clf.fit(X_train, y_train)
        accuracy = self.clf.score(X_test, y_test)
        return accuracy

    def stock_predict(self):
        forecast_set = self.clf.predict(self.X_lately)
        return forecast_set

    def stock_predict_plot(self, df, forecast_set):
        df['Forecast'] = np.nan
        last_date = df.iloc[-1].name
        last_unix = last_date.timestamp()
        one_day = 86400
        next_unix = last_unix + one_day
        for i in forecast_set:
            next_date = datetime.datetime.fromtimestamp(next_unix)
            next_unix += one_day
            df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

        df['Adj. Close'].plot()
        df['Forecast'].plot()
        plt.legend(loc=4)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.show()