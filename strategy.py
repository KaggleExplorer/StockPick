from sys import stdout
import numpy as np
import pandas as pd
from pandas_datareader import data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.metrics import classification_report
from tqdm import tqdm
import utils


class TickerProvider:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = utils.config_provider(self.config_file)
        self.api_key = self.config['api_key']
        self.ticker_url = utils.attach_api_key_to_url(self.config['ticker_url'], self.config['api_key'])
        self.profile_url = self.config['profile_url']

    def get_all_tickers(self, from_file=True, debug=True):
        if from_file:
            df = pd.read_csv('./data/symbols.csv')
            ticker_list = df['symbol'].to_list()
            if debug:
                return ticker_list[:20]
            return ticker_list
        else:
            tickers_json = utils.get_json_from_url(self.ticker_url)
            df = pd.DataFrame.from_records(tickers_json['symbolsList'])
            return df['symbol'].to_list()

    def get_ticker_by_sector(self, sector, from_file=True):
        if from_file:
            df = pd.read_csv('./data/sectors.csv', index_col='Symbol')
            tickers_selected = df[df['Sector'] == sector].index.values.tolist()
            return tickers_selected
        else:
            tickers_sector = []
            available_tickers = self.get_all_tickers()
            for t in tqdm(available_tickers):
                url = utils.attach_api_key_to_url(self.profile_url + t, self.api_key)
                d = utils.get_json_from_url(url)
                tickers_sector.append(utils.find_in_json(d, 'sector'))

            df = pd.DataFrame(tickers_sector, index=available_tickers, columns=['Sector'])
            tickers_selected = df[df['Sector'] == sector].index.values.tolist()
            return tickers_selected

    def tickers_to_csv(self):
        tickers_json = utils.get_json_from_url(self.ticker_url)
        df = pd.DataFrame.from_records(tickers_json['symbolsList'])
        df.to_csv('./data/symbols.csv', index=False)

    def tickers_by_sectors_to_csv(self):
        tickers_sector = []
        available_tickers = self.get_all_tickers(from_file=True, debug=False)

        for t in tqdm(available_tickers):
            url = utils.attach_api_key_to_url(self.profile_url + t, self.api_key)
            d = utils.get_json_from_url(url)
            tickers_sector.append(d['profile']['sector'])

        df = pd.DataFrame(tickers_sector, index=available_tickers, columns=['Sector'])
        df.index.names = ['Symbol']
        df.to_csv('./data/sectors.csv')


def get_price_change_percent(symbol, start, end, src='yahoo', field='Adj Close'):
    prices = data.DataReader(symbol, src, start, end)[field]
    last, first = prices.index[-1], prices.index[0]
    price_change = (prices[last] - prices[first]) / prices[first] * 100
    return price_change


if __name__ == '__main__':
    tp = TickerProvider('./config.json')
    print(tp.get_all_tickers(from_file=True))
    print(tp.get_ticker_by_sector('Technology', from_file=True))
    # print(tp.get_ticker_by_sector('Technology'))
    # tp.tickers_to_csv()
    # tp.tickers_by_sectors_to_csv()
    # print(get_price_change_percent('NVDA', '2019-01-01', '2019-12-31'))

