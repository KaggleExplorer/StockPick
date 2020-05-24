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

    def get_ticker_by_sector(self, sector, from_file=True, debug=True):
        if from_file:
            df = pd.read_csv('./data/sectors.csv', index_col='Symbol')
            tickers_selected = df[df['Sector'] == sector].index.values.tolist()
            if debug:
                return tickers_selected[:20]
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


class PerformanceProvider:
    def __init__(self, tickers, start, end, config_file):
        self.tickers = tickers
        self.start = start
        self.end = end
        self.indicators = pd.read_csv('./data/indicators.csv')['indicator'].to_list()
        self.config = utils.config_provider(config_file)
        self.api_key = self.config['api_key']
        self.income_statement_url = self.config['income_statement_url']
        self.balance_sheet_statement_url = self.config['balance_sheet_statement_url']
        self.cash_flow_statement_url = self.config['cash_flow_statement_url']
        self.financial_ratios_url = self.config['financial_ratios_url']
        self.company_key_metrics_url = self.config['company_key_metrics_url']
        self.financial_statement_growth_url = self.config['financial_statement_growth_url']

    @staticmethod
    def get_price_change_percent(symbol, start, end, src='yahoo', field='Adj Close'):
        try:
            prices = data.DataReader(symbol, src, start, end)[field]
            last, first = prices.index[-1], prices.index[0]
            price_change = (prices[last] - prices[first]) / prices[first] * 100
            return price_change
        except:
            return 0

    def populate_price_change(self):
        df = pd.DataFrame(data={'Symbol': self.tickers})
        df['PercentPriceChange'] = df['Symbol'].apply(lambda r: self.get_price_change_percent(r, self.start, self.end))
        return df

    def populate_financial_indicators(self, year):
        # Initialize lists and dataframe (dataframe is a 2D numpy array filled with 0s)
        missing_tickers, missing_index = [], []
        d = np.zeros((len(self.tickers), len(self.indicators)))

        for t, _ in enumerate(tqdm(self.tickers)):
            url0 = utils.attach_api_key_to_url(self.income_statement_url + self.tickers[t], self.api_key)
            url1 = utils.attach_api_key_to_url(self.balance_sheet_statement_url + self.tickers[t], self.api_key)
            url2 = utils.attach_api_key_to_url(self.cash_flow_statement_url + self.tickers[t], self.api_key)
            url3 = utils.attach_api_key_to_url(self.financial_ratios_url + self.tickers[t], self.api_key)
            url4 = utils.attach_api_key_to_url(self.company_key_metrics_url + self.tickers[t], self.api_key)
            url5 = utils.attach_api_key_to_url(self.financial_statement_growth_url + self.tickers[t], self.api_key)
            a0 = utils.get_json_from_url(url0)
            a1 = utils.get_json_from_url(url1)
            a2 = utils.get_json_from_url(url2)
            a3 = utils.get_json_from_url(url3)
            a4 = utils.get_json_from_url(url4)
            a5 = utils.get_json_from_url(url5)

            # Combine all json files in a list, so that it can be scanned quickly
            combined = [a0, a1, a2, a3, a4, a5]
            all_dates = utils.find_in_json(combined, 'date')

            check = [s for s in all_dates if year in s]  # find all 2018 entries in dates
            if len(check) > 0:
                date_index = all_dates.index(check[0])  # get most recent 2018 entries, if more are present

                for i, _ in enumerate(self.indicators):
                    ind_list = utils.find_in_json(combined, self.indicators[i])
                    try:
                        d[t][i] = ind_list[date_index]
                    except:
                        d[t][i] = np.nan  # in case there is no value inserted for the given indicator

            else:
                missing_tickers.append(self.tickers[t])
                missing_index.append(t)

        actual_tickers = [x for x in self.tickers if x not in missing_tickers]
        d = np.delete(d, missing_index, 0)
        # raw dataset
        raw_data = pd.DataFrame(d, index=actual_tickers, columns=self.indicators)
        return raw_data


if __name__ == '__main__':
    tp = TickerProvider('./config.json')
    # print(tp.get_all_tickers(from_file=True))
    # print(tp.get_ticker_by_sector('Technology', from_file=True))
    tech_tickers = tp.get_ticker_by_sector('Technology', from_file=True, debug=True)
    pp = PerformanceProvider(tech_tickers, '2019-01-02', '2019-12-31', './config.json')
    df = pp.populate_price_change()
    raw_df = pp.populate_financial_indicators('2018')
    print(df)
    print(raw_df)
    # print(tp.get_ticker_by_sector('Technology'))
    # tp.tickers_to_csv()
    # tp.tickers_by_sectors_to_csv()
    # print(get_price_change_percent('NVDA', '2019-01-01', '2019-12-31'))

