# Importamos las librerías necesarias.
import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor
from matplotlib import pyplot
from fbprophet.diagnostics import performance_metrics, cross_validation
from fbprophet.plot import plot_cross_validation_metric
import itertools
import numpy as np
import seaborn as sns

# Abrimos el archivo.

file_path = '/home/rafaelfp/Dropbox/Postgrados/MDS/MLA/Tarea MLA/data/Tarea MLA - Hoja 2.csv'
df = pd.read_csv(file_path)

# Le doy formato a la fecha para poder trabajarla.
df['ds'] = pd.to_datetime(df['ds'], format='%d/%m/%Y')
df = df.sort_values(by=['ds'], ascending=True)
df.reset_index(drop=True, inplace=True)

# Guardo el df ordenado para usarlo en el pronóstico del modelo mejor.
df.to_csv(r'/home/rafaelfp/Dropbox/Postgrados/MDS/MLA/Tarea MLA/data/datos2.csv',
          index=False)

path2 = '/home/rafaelfp/Dropbox/Postgrados/MDS/MLA/Tarea MLA/data/datos2.csv'

# Graficamos los datos.
graf = sns.histplot(df, kde='true')
plt.show()

outliers = sns.boxplot(df['y'])
plt.show()

# Le indicamos al modelos los días "importantes"
dias_especiales = pd.DataFrame({
    'holiday': 'dias_especiales',
    'ds': pd.to_datetime(['2018-02-14', '2019-02-14', '2020-02-14',
                          '2021-02-14', '2022-02-14'], format='%Y-%m-%d'),
    'lower_window': -1,
    'upper_window': 0,
})
holidays = dias_especiales

# Creamos el modelo Prophet y le hacemos un fit.
m = Prophet(holidays=holidays, weekly_seasonality=True)
m.add_country_holidays(country_name='Chile')
m.fit(df)

# Se indica cuáles serán los futures.
future = m.make_future_dataframe(periods=7)
future.tail()

# Forecast
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# Se grafican los componentes del forecast (trend, weekly, yearly)
fig2 = m.plot_components(forecast)
plt.show()

# Se grafica cuándo se producen los mayores cambios en la tendencia.
fig3 = m.plot(forecast)
a = add_changepoints_to_plot(fig3.gca(), m, forecast)
plt.show()

# Crossvalidation.
df_cv = cross_validation(m, initial='120 days', horizon='7 days',
                         parallel='processes')

df_p = performance_metrics(df_cv, rolling_window=1)
print(df_p.head())

fig4 = plot_cross_validation_metric(df_cv, metric='mape')
plt.show()

# Ahora pruebo varios hiperparámetros para correr el modelo denuevo y
# compararlo.
param_grid = {
    'changepoint_prior_scale': [0.5, 0.8, 1, 2, 3, 4, 5, 6],  #, 0.6, 0.7, 0.8, 0.9],

    'changepoint_range': [0.9, 0.95, 0.99],
}

# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v))
              for v in itertools.product(*param_grid.values())]
mapes = []  # Store the mapes for each params here

# Use cross validation to evaluate all parameters
for params in all_params:
    m = Prophet(**params).fit(df)  # Fit model with given params
    df_cv = cross_validation(m, initial='120 days', horizon='7 days',
                             parallel='processes')
    df_p = performance_metrics(df_cv, rolling_window=1)
    mapes.append(df_p['mape'].values[0])

# Find the best parameters
tuning_results = pd.DataFrame(all_params)
tuning_results['mae'] = mapes


# Se imprime el mejor parámetro.
best_params = all_params[np.argmin(mapes)]

# Creamos el modelo Prophet con el nuevo hiperparámetro 0.9 y 0.95 y
# le hacemos un fit.
m2 = Prophet(changepoint_prior_scale=0.5, changepoint_range=0.9,
             weekly_seasonality=True, holidays=holidays)
m2.add_country_holidays(country_name='Chile')
m2.fit(df)

# Se indica cuáles serán los futures y el período hacia adelante.
future2 = m2.make_future_dataframe(periods=7)
future2.tail()

# Forecast
forecast2 = m2.predict(future2)
forecast2[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# Se grafican los componentes del forecast (trend, weekly, yearly)
fig6 = m2.plot_components(forecast2)
plt.show()

# Se grafica cuándo se producen los mayores cambios en la tendencia.
fig7 = m2.plot(forecast2)
a2 = add_changepoints_to_plot(fig7.gca(), m2, forecast2)
plt.show()

df_cv2 = cross_validation(m2, initial='120 days', horizon='7 days',
                          parallel='processes')

df_p2 = performance_metrics(df_cv2, rolling_window=1)

fig8 = plot_cross_validation_metric(df_cv2, metric='mape')
plt.show()

print('MAPE de Prophet: %.3f' % df_p2['mape'])


###############################################################################
# XGBoost.
###############################################################################
m2 = XGBRegressor()


# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values


# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test, :], data[-n_test:, :]


# fit an xgboost model and make a one step prediction
def xgboost_forecast(train, testX):
    # transform list into array
    train = asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=3000,
                         max_depth=20, booster='gbtree')
    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict(asarray([testX]))
    return yhat[0]


# walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # split test row into input and output columns
        testX, testy = test[i, :-1], test[i, -1]
        # fit model on history and make a prediction
        yhat = xgboost_forecast(history, testX)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
        # summarize progress
        print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
    # estimate prediction error
    error = mean_absolute_percentage_error(test[:, -1], predictions)
    return error, test[:, -1], predictions


# forecast monthly selling with xgboost

# load the dataset
series = read_csv(path2, header=0, index_col=0)
values = series.values

# transform the time series data into supervised learning
data = series_to_supervised(values, n_in=6)

# evaluate
mape, y, yhat = walk_forward_validation(data, 30)

# plot expected vs predicted
pyplot.plot(y, label='Expected')
pyplot.plot(yhat, label='Predicted')
pyplot.legend()
pyplot.show()

print('MAPE de Prophet: %.3f' % df_p2['mape'])
print('MAPE de XGBoost: %.3f' % mape)

# Pronosticando próximo día tomando en cuenta los 10 días anteriores.
# load the dataset
series = read_csv(path2, header=0, index_col=0)
values = series.values

# transform the time series data into supervised learning
train = series_to_supervised(values, n_in=4)

# split into input and output columns
trainX, trainy = train[:, :-1], train[:, -1]

# fit model
model = XGBRegressor(objective='reg:squarederror', n_estimators=3000,
                     max_depth=20, booster='gbtree')
model.fit(trainX, trainy)

# construct an input for a new prediction
row = values[-4:].flatten()

# make a one-step prediction
yhat = model.predict(asarray([row]))
print('Input: %s, Predicted with XGBoost: %.3f' % (row, yhat[0]))

# TODO: Implementar LightGBM
# TODO: Leer bien sobre changepoint_range=0.90
