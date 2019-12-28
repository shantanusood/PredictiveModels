from com.silpaservices.ml.algos.StockData import StockData

if __name__ == '__main__':
    s = StockData('AAPL')
    df = s.get_stock_data()
    s.stock_test()
    x = s.stock_predict()
    s.stock_predict_plot(df, x)
