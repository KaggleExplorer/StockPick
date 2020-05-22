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


config = utils.config_provider('./config.json')
api_key = config['api_key']
ticker_url = utils.attach_api_key_to_url(config['ticker_url'], config['api_key'])
profile_url = config['profile_url']


def get_all_tickers(url):
    return ['SPY', 'CMCSA', 'KMI', 'INTC', 'MU', 'GDX', 'GE', 'BAC', 'EEM', 'XLF', 'AAPL', 'MSFT', 'SIRI', 'HPQ']
    ticks_json = utils.get_json_from_url(url)
    available_tickers = utils.find_in_json(ticks_json, 'symbol')
    return available_tickers


def get_ticker_sector_info(sector):
    tickers_sector = []
    available_tickers = get_all_tickers(ticker_url)
    for t in tqdm(available_tickers):
        url = utils.attach_api_key_to_url(profile_url + t, api_key)
        d = utils.get_json_from_url(url)
        tickers_sector.append(utils.find_in_json(d, 'sector'))

    df = pd.DataFrame(tickers_sector, index=available_tickers, columns=['Sector'])
    tickers_tech = df[df['Sector'] == sector].index.values.tolist()
    return tickers_tech


if __name__ == '__main__':
    print(get_all_tickers(ticker_url))
    print(get_ticker_sector_info('Technology'))

