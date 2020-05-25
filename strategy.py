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
                return ticker_list[:200]
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
                return tickers_selected[:200]
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
        self._price_change = None

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
        df.set_index('Symbol', inplace=True)
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

    @property
    def price_change(self):
        if self._price_change is not None and not self._price_change.empty:
            return self._price_change
        self._price_change = self.populate_price_change()
        return self._price_change

    def combine_and_clean(self, year):
        combined_data = self.populate_financial_indicators(year)
        # Remove columns that have more than 20 0-values
        combined_data = combined_data.loc[:, combined_data.isin([0]).sum() <= 20]

        # Remove columns that have more than 15 nan-values
        combined_data = combined_data.loc[:, combined_data.isna().sum() <= 15]

        # Fill remaining nan-values with column mean value
        combined_data = combined_data.apply(lambda x: x.fillna(x.mean()))

        # Get price variation data only for tickers to be used
        filtered_pc = self.price_change.loc[combined_data.index.values, :]
        filtered_pc['Class'] = filtered_pc['PercentPriceChange'].apply(lambda r: 1 if r >= 0 else 0)
        combined_data['Class'] = filtered_pc['Class']
        return combined_data


class DataProcessor:
    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.split()
        self.scale()

    def split(self):
        self._train, self._test = train_test_split(self.raw_data, test_size=0.2, random_state=1, stratify=self.raw_data['Class'])
        self._X_train, self._y_train = self._train.iloc[:, :-1].values, self._train.iloc[:, -1].values
        self._X_test, self._y_test = self._test.iloc[:, :-1].values, self._test.iloc[:, -1].values
        print(f'Number of training samples: {self._X_train.shape[0]}')
        print(f'Number of testing samples: {self._X_test.shape[0]}')
        print(f'Number of features: {self._X_train.shape[1]}')

    def scale(self):
        scaler = StandardScaler()
        scaler.fit(self._X_train)
        self._X_train = scaler.transform(self._X_train)
        self._X_test = scaler.transform(self._X_test)

    @property
    def X_train(self):
        return self._X_train

    @property
    def X_test(self):
        return self._X_test

    @property
    def y_train(self):
        return self._y_train

    @property
    def y_test(self):
        return self._y_test

    @property
    def train_set(self):
        return self._train

    @property
    def test_set(self):
        return self._test


class Models:
    def __init__(self, data_processor):
        self.data_processor = data_processor

    @property
    def svm(self):
        return self.model_svm()

    @property
    def rf(self):
        return self.model_rf()

    @property
    def xgb(self):
        return self.model_xgb()

    @property
    def mlp(self):
        return self.model_mlp()

    def model_svm(self):  # support vector machine
        tuned_parameters = [{'kernel': ['rbf', 'linear'],
                             'gamma': [1e-3, 1e-4],
                             'C': [0.01, 0.1, 1, 10, 100]}]

        clf1 = GridSearchCV(SVC(random_state=1),
                            tuned_parameters,
                            n_jobs=6,
                            scoring='precision_weighted',
                            cv=5)
        clf1.fit(self.data_processor.X_train, self.data_processor.y_train)
        return clf1

        # print('Best score and parameters found on development set:')
        # print('%0.3f for %r' % (clf1.best_score_, clf1.best_params_))

    def model_rf(self):  # random forest
        tuned_parameters = {'n_estimators': [32, 256, 512, 1024],
                            'max_features': ['auto', 'sqrt'],
                            'max_depth': [4, 5, 6, 7, 8],
                            'criterion': ['gini', 'entropy']}
        clf2 = GridSearchCV(RandomForestClassifier(random_state=1),
                            tuned_parameters,
                            n_jobs=6,
                            scoring='precision_weighted',
                            cv=5)
        clf2.fit(self.data_processor.X_train, self.data_processor.y_train)
        return clf2
        # print('Best score and parameters found on development set:')
        # print('%0.3f for %r' % (clf2.best_score_, clf2.best_params_))

    def model_xgb(self):  # extreme gradient boosting
        tuned_parameters = {'learning_rate': [0.01, 0.001],
                            'max_depth': [4, 5, 6, 7, 8],
                            'n_estimators': [32, 128, 256]}
        clf3 = GridSearchCV(xgb.XGBClassifier(random_state=1),
                            tuned_parameters,
                            n_jobs=6,
                            scoring='precision_weighted',
                            cv=5)
        clf3.fit(self.data_processor.X_train, self.data_processor.y_train)
        return clf3
        # print('Best score and parameters found on development set:')
        # print('%0.3f for %r' % (clf3.best_score_, clf3.best_params_))

    def model_mlp(self):  # multi layer perception
        tuned_parameters = {'hidden_layer_sizes': [(32,), (64,), (32, 64, 32)],
                            'activation': ['tanh', 'relu'],
                            'solver': ['lbfgs', 'adam']}
        clf4 = GridSearchCV(MLPClassifier(random_state=1, batch_size=4, early_stopping=True),
                            tuned_parameters,
                            n_jobs=6,
                            scoring='precision_weighted',
                            cv=5)
        clf4.fit(self.data_processor.X_train, self.data_processor.y_train)
        return clf4
        # print('Best score, and parameters, found on development set:')
        # print('%0.3f for %r' % (clf4.best_score_, clf4.best_params_))


def model_evaluator(model, data_processor, performance_provider):
    # Initial investment can be $100 for each stock whose predicted class = 1
    buy_amount = 100
    pvar_test = performance_provider.price_change.loc[data_processor.test_set.index.values, :]

    # In new dataframe df, store all the information regarding each model's predicted class and relative pnl in $USD
    # first column is the true class (BUY/INGORE)
    df = pd.DataFrame(data_processor.y_test, index=data_processor.test_set.index.values, columns=['ACTUAL'])
    df['SVM'] = model.svm.predict(data_processor.X_test)  # predict class for testing dataset
    df['VALUE START SVM [$]'] = df['SVM'] * buy_amount  # if class = 1 --> buy $100 of that stock
    df['VAR SVM [$]'] = (pvar_test['PercentPriceChange'].values / 100) * df['VALUE START SVM [$]']
    df['VALUE END SVM [$]'] = df['VALUE START SVM [$]'] + df['VAR SVM [$]']  # compute final value
    df['RF'] = model.rf.predict(data_processor.X_test)
    df['VALUE START RF [$]'] = df['RF'] * buy_amount
    df['VAR RF [$]'] = (pvar_test['PercentPriceChange'].values / 100) * df['VALUE START RF [$]']
    df['VALUE END RF [$]'] = df['VALUE START RF [$]'] + df['VAR RF [$]']
    df['XGB'] = model.xgb.predict(data_processor.X_test)
    df['VALUE START XGB [$]'] = df['XGB'] * buy_amount
    df['VAR XGB [$]'] = (pvar_test['PercentPriceChange'].values / 100) * df['VALUE START XGB [$]']
    df['VALUE END XGB [$]'] = df['VALUE START XGB [$]'] + df['VAR XGB [$]']
    df['MLP'] = model.mlp.predict(data_processor.X_test)
    df['VALUE START MLP [$]'] = df['MLP'] * buy_amount
    df['VAR MLP [$]'] = (pvar_test['PercentPriceChange'].values / 100) * df['VALUE START MLP [$]']
    df['VALUE END MLP [$]'] = df['VALUE START MLP [$]'] + df['VAR MLP [$]']
    return df


if __name__ == '__main__':
    tp = TickerProvider('./config.json')
    # print(tp.get_all_tickers(from_file=True))
    # print(tp.get_ticker_by_sector('Technology', from_file=True))
    tech_tickers = tp.get_ticker_by_sector('Technology', from_file=True, debug=True)
    pp = PerformanceProvider(tech_tickers, '2019-01-02', '2019-12-31', './config.json')
    df = pp.populate_price_change()
    combined_df = pp.combine_and_clean('2018')
    data_processor = DataProcessor(combined_df)
    models = Models(data_processor)
    eva = model_evaluator(models, data_processor, pp)
    print(df)
    print(combined_df)
    print(eva)
    # print(tp.get_ticker_by_sector('Technology'))
    # tp.tickers_to_csv()
    # tp.tickers_by_sectors_to_csv()
    # print(get_price_change_percent('NVDA', '2019-01-01', '2019-12-31'))

