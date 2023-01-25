import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web
import yfinance as yf
import datetime
import scipy
import warnings
import optuna
import plotly

warnings.filterwarnings('ignore')


def exponentially_weighted_mean(prices, window, alpha):
    # ema = [prices[:ema_window].sum() / ema_window]
    ema = [prices[:window].mean()]
    for price in prices[window:]:
        ema.append((price * alpha) + ema[-1] * (1 - alpha))

    ema_series = pd.Series(ema, index=prices[window - 1:].index, name='EMA')
    return ema_series


def exponentially_weighted_variance(prices, window, alpha):
    ewv = [prices[:window].var()]
    ema = exponentially_weighted_mean(prices, window - 1, alpha)
    for k, price in enumerate(prices[window:]):
        ewv.append((1 - alpha) * (ewv[-1] + alpha * (price - ema[k]) ** 2))
    ewv = np.sqrt(ewv)
    ema_std_series = pd.Series(ewv, index=prices[window - 1:].index, name='EMA_STD')
    return ema_std_series


def ema_reversion(prices, ema_window, ema_std_window, ema_alpha, ema_std_alpha, short, long):
    df = prices.to_frame(name='Close')

    ema_series = exponentially_weighted_mean(df['Close'], ema_window, ema_alpha)

    df = df.merge(ema_series, how='left', on='Date')
    ema_std_series = exponentially_weighted_variance(df['Close'], ema_std_window, ema_std_alpha)

    df = df.merge(ema_std_series, how='left', on='Date')
    # print(df['STD'])

    df['BOLU'] = df['EMA'] + short * df['EMA_STD']
    df['BOLD'] = df['EMA'] - long * df['EMA_STD']
    df['ZSCORE'] = (df['Close'] - df['EMA']) / df['EMA_STD']
    # print(df[:15].to_string())
    df = df.dropna()  # Remove missing values
    # print(df)

    # plt.show()
    df['LONG'] = df['ZSCORE'] < -long
    df['SHORT'] = df['ZSCORE'] > short
    # df['FLAT'] = abs(df['ZSCORE']) <

    df['POSITIONS'] = 0
    df['POSITIONS'][df['LONG']] = 1
    df['POSITIONS'][df['SHORT']] = -1
    df['POSITIONS'] = df['POSITIONS'] * df['Close']
    ######################################
    # df[['BOLU', 'BOLD', 'Close', 'EMA']].plot(color=['red', 'blue', 'purple', 'green'])
    # df['Close'][df['LONG']].plot(linestyle='none', color='black', marker='^')
    # df['Close'][df['SHORT']].plot(linestyle='none', color='black', marker='o')
    # plt.show()
    ######################################
    # df['POSITIONS'] = df['POSITIONS'].fillna(method='ffill')
    # df['POSITIONS'] = df['POSITIONS'].fillna(method='bfill')
    # df['PNL0'] = df['POSITIONS'].shift()*df['Close'].diff()/df['Close'].shift()
    df['PNL'] = df['POSITIONS'].shift() * df[
        'Close'].pct_change()  # pct_change: Percentage change between the current and a prior element
    df['Ret'] = df['PNL'] / np.abs(df['POSITIONS'].shift())
    Sharpe = np.sqrt(252) * df['Ret'].mean() / df['Ret'].std()
    APR = np.prod(1 + df['Ret']) ** (252 / len(df['Ret'])) - 1

    return APR, Sharpe

    # print(df.to_string())
    # print(Sharpe)
    # plt.show()


def ema_reversion_backtesting(ema_window, ema_std_window, ema_alpha, ema_std_alpha, short, long):
    symbol = 'AAPL'
    endDate = datetime.datetime.now()
    startDate = endDate - datetime.timedelta(days=365)
    data = yf.Ticker(symbol)
    df = data.history(start=startDate, end=endDate)
    n = 100
    syn_series = pd.DataFrame(df['Close'])
    for i in range(n):
        [mean, std] = scipy.stats.norm.fit(df['Close'].diff().dropna())
        innovation = scipy.stats.norm.rvs(loc=mean, scale=std, size=df.shape[0]-1)

        cum_sum = syn_series.iloc[0,0]
        syn_data = [cum_sum]
        for value in innovation:
            cum_sum = cum_sum + value
            syn_data.append(cum_sum)
        syn_data = pd.DataFrame(syn_data, columns=['Close' + str(i)], index=df.index)
        syn_series = syn_series.merge(syn_data, how='left', on='Date')
    Sharpe_list = []
    APR_list = []
    for i in range(n):
        close_prices = syn_series['Close' + str(i)]
        APR, Sharpe = ema_reversion(close_prices, ema_window, ema_std_window, ema_alpha, ema_std_alpha, short, long)
        APR_list.append(APR)
        Sharpe_list.append(Sharpe)
    return APR_list, Sharpe_list


def objective(trial):
    ema_window = trial.suggest_int('ema_window', 2, 180)
    ema_std_window = trial.suggest_int('ema_std_window', 2, 180)
    ema_alpha = trial.suggest_float('ema_alpha', 0, 1)
    ema_std_alpha = trial.suggest_float('ema_std_alpha', 0, 1)
    short = trial.suggest_float('short', 0, 5)
    long = trial.suggest_float('long', 0, 5)
    APR_list, Sharpe_list = ema_reversion_backtesting(ema_window, ema_std_window, ema_alpha, ema_std_alpha, short, long)
    return np.mean(APR_list), np.mean(Sharpe_list)


if __name__ == '__main__':
    # To start the optimization, we create a study object and pass the objective function to method
    study = optuna.create_study(directions=["maximize", "maximize"], study_name='example_study1', storage='sqlite:///example.db', load_if_exists=True)
    study.optimize(objective, n_jobs=-1, n_trials=100)
    # You can get the best parameter as follows.
    print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

    trial_with_highest_accuracy = max(study.best_trials, key=lambda t: t.values[1])
    print(f"Trial with highest accuracy: ")
    print(f"\tnumber: {trial_with_highest_accuracy.number}")
    print(f"\tparams: {trial_with_highest_accuracy.params}")
    print(f"\tvalues: {trial_with_highest_accuracy.values}")

    optuna.visualization.plot_pareto_front(study, target_names=['APR', 'Sharpe Ratio'])
    plt.show()
    # print(f"Mejores parámetros a: {best_params['a']}, c: {best_params['c']}, índice de Sharpe: {study.best_value}")
    # optuna.visualization.plot_optimization_history(study).show()

    # print("Number of finished trials: ", len(study.trials))

    # symbol = 'ADA-USD'
    # # startDate = datetime.datetime(2022, 1, 1)
    # # endDate = datetime.datetime(2022, 12, 31)
    # ema_window = 50
    # ema_std_window = 20
    # ema_alpha = 0.05
    # ema_std_alpha = 0.05
    # long = 0.8
    # short = 0.8
    # data = yf.Ticker(symbol)
    # # df = data.history(start=startDate, end=endDate)
    # APR, SharpeRatio = ema_reversion(symbol, ema_window, ema_std_window, ema_alpha, ema_std_alpha, short, long)
    # print('APR=%f Sharpe=%f' % (APR, SharpeRatio))
    # # prices = df['Close']
    # # # m = prices[:5].sum()/5
    # # # print(type(df['Close'].values))
    # # g = df[3:9]['Close']
    # # f = pd.Series([8, 5, 4, 3, 2, 1], index=g.index)
    # # h = [8, 5, 4, 3, 2]
    # # h=pd.Series(h, index= g[1:].index, name='FFFF')
    # # g = pd.merge(g, h, how='left', on='Date') #, right_index=True, left_index=True)
    # # print(g)
    # # m = exponentially_weighted_mean(prices, 5, 0.3)
    # # h = exponentially_weighted_variance(prices, 8, 0.5)
    # # # print(prices)
    # # print(m)
    # # print(h)
