###############################################################################
# Importamos las librerías necesarias.
###############################################################################
import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from matplotlib import pyplot
from fbprophet.diagnostics import performance_metrics, cross_validation
from fbprophet.plot import plot_cross_validation_metric
import itertools
import numpy as np
import seaborn as sns
import random
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold

random.seed(1234)

###############################################################################
# IO
###############################################################################


file_path = '/home/rafaelfarias/Dropbox/Postgrados/MDS/MLA/Tarea MLA/data/Tarea MLA - Hoja 2.csv'
df = pd.read_csv(file_path)

# Le doy formato a la fecha para poder trabajarla.
df['ds'] = pd.to_datetime(df['ds'], format='%d/%m/%Y')
df = df.sort_values(by=['ds'], ascending=True)
df.reset_index(drop=True, inplace=True)

# Guardo el df ordenado para usarlo en el pronóstico del modelo mejor.
df.to_csv(
    r'/home/rafaelfarias/Dropbox/Postgrados/MDS/MLA/Tarea MLA/data/datos2.csv',
    index=False)

path2 = '/home/rafaelfarias/Dropbox/Postgrados/MDS/MLA/Tarea MLA/data/datos2.csv'

###############################################################################
# Estadística descriptiva.
###############################################################################

# Graficamos los datos.
graf = sns.histplot(data=df, kde='true')
plt.show()

outliers = sns.boxplot(data=df['y'])
plt.show()

# Describo los datos.
describe = df.describe()

# Le indicamos al modelos los días "importantes"
dias_especiales = pd.DataFrame({
    'holiday': 'dias_especiales',
    'ds': pd.to_datetime(['2018-02-14', '2019-02-14', '2020-02-14',
                          '2021-02-14', '2022-02-14'], format='%Y-%m-%d'),
    'lower_window': -1,
    'upper_window': 1,
})
holidays = dias_especiales

###############################################################################
# Modelo Prophet sin tuning.
###############################################################################

# Creamos el modelo Prophet y le hacemos un fit.
m = Prophet(holidays=holidays, weekly_seasonality=True,
            daily_seasonality=False,
            yearly_seasonality=False, n_changepoints=20)
m.add_country_holidays(country_name='Chile')
m.fit(df)

# Se indica cuáles serán los futures.
future = m.make_future_dataframe(periods=7)
future.tail()

# Forecast
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(14)

# Se grafican los componentes del forecast (trend, weekly, yearly)
fig2 = m.plot_components(forecast)
plt.title('Componentes del forecast sin tuning')
plt.show()

# Se grafica cuándo se producen los mayores cambios en la tendencia.
fig3 = m.plot(forecast)
a = add_changepoints_to_plot(fig3.gca(), m, forecast)
plt.title('Pronóstico con changepoints modelo sin tunear')
plt.show()

# Crossvalidation.
df_cv = cross_validation(m, initial='30 days', horizon='7 days',
                         parallel='processes', period='1 days')

df_p = performance_metrics(df_cv, rolling_window=1)

fig4 = plot_cross_validation_metric(df_cv, metric='mae')
plt.title('MAE del modelo sin tuning')
plt.show()

print('MAE de Prophet sin tuning: %.3f' % df_p['mae'])

###############################################################################
# Hiperparámetros
###############################################################################

# param_grid = {
#     'changepoint_prior_scale': [0.4, 0.5],
#
#     'changepoint_range': [0.8, 0.9],
#
#     'seasonality_prior_scale': [0.001, 0.01, 0.1, 0.5, 0.7, 1, 5, 10],
#
#     'holidays_prior_scale': [0.001, 0.01, 0.1, 0.5, 0.7, 1, 5, 10]
# }
#
# # Generate all combinations of parameters
# all_params = [dict(zip(param_grid.keys(), v))
#               for v in itertools.product(*param_grid.values())]
# maes = []  # Store the maes for each params here
#
# # Use cross validation to evaluate all parameters
# for params in all_params:
#     m = Prophet(**params, yearly_seasonality=False, daily_seasonality=False,
#                 weekly_seasonality=True,
#                 holidays=holidays, n_changepoints=20)
#     m.add_country_holidays(country_name='Chile')
#     m.fit(df)  # Fit model with given params
#     df_cv = cross_validation(m, initial='30 days', horizon='7 days',
#                              parallel='processes', period='1 days')
#     df_p = performance_metrics(df_cv, rolling_window=1)
#     maes.append(df_p['mae'].values[0])
#
# # Find the best parameters
# tuning_results = pd.DataFrame(all_params)
# tuning_results['mae'] = maes
#
# # Se imprime el mejor parámetro.
# best_params = all_params[np.argmin(maes)]
# best_params
# Darle index para lograr automatizar el paso de más abajo

###############################################################################
# Modelo Prophet con hiperparámetros.
###############################################################################

m2 = Prophet(changepoint_prior_scale=0.5, changepoint_range=0.9,
             seasonality_prior_scale=0.1, holidays_prior_scale=0.01,
             weekly_seasonality=True, holidays=holidays,
             yearly_seasonality=False, daily_seasonality=False,
             n_changepoints=22, interval_width=0.8)
m2.add_country_holidays(country_name='Chile')
m2.fit(df)

# Se indica cuáles serán los futures y el período hacia adelante.
future2 = m2.make_future_dataframe(periods=7)
future2.tail()

# Forecast
forecast2 = m2.predict(future2)
forecast2[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(14)

forecast2[['yhat_upper']].sum()

# Se grafican los componentes del forecast (trend, weekly, yearly)
fig6 = m2.plot_components(forecast2)
plt.title('Componentes del forecast con tuning')
plt.show()

# Se grafica cuándo se producen los mayores cambios en la tendencia.
fig7 = m2.plot(forecast2)
a2 = add_changepoints_to_plot(fig7.gca(), m2, forecast2)
plt.title('Pronóstico con changepoints incluidos modelo tuneado')
plt.show()

# Crossvalidation
df_cv2 = cross_validation(m2, initial='30 days', horizon='7 days',
                          parallel='processes', period='1 days')

df_p2 = performance_metrics(df_cv2, rolling_window=1)

fig8 = plot_cross_validation_metric(df_cv2, metric='mae')
plt.title('MAE del modelo tuneado')
plt.show()

print('MAE de Prophet tuneado: %.3f' % df_p2['mae'])


###############################################################################
# XGBoost 1 sin tuning.
###############################################################################
# Series a aprendizaje supervisado.
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


# train/test sets
def train_test_split(data, n_test):
    return data[:-n_test, :], data[-n_test:, :]


# Fit y un paso de predicción.
def xgboost_forecast(train, testX):
    # transformando listas en arreglos.
    train = asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit al modelo.
    model = XGBRegressor(objective='reg:squarederror', max_depth=2)
    model.fit(trainX, trainy)
    # un paso de predicción.
    yhat = model.predict(asarray([testX]))
    return yhat[0]


# Walk-forward validación para la data sin variaciones.
def walk_forward_validation(data, n_test):
    predictions = list()
    # separando el dataset
    train, test = train_test_split(data, n_test)
    # Creando history con los datos de entrenamiento.
    history = [x for x in train]
    # Pasos de cada time-step el set de testeo.
    for i in range(len(test)):
        # separar el testeo entre input y output
        testX, testy = test[i, :-1], test[i, -1]
        # Fit al modelo en History y haciendo una predicción.
        yhat = xgboost_forecast(history, testX)
        # Guardando el forecast en una lista de predicciones.
        predictions.append(yhat)
        # agregando la obervación actual de history al siguiente loop.
        history.append(test[i])
        # Resumiendo el progreso.
        print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
    # Estimando el error de la predicción.
    error = mean_absolute_error(test[:, -1], predictions)
    return error, test[:, -1], predictions


# Pronosticando próximo día tomando en cuenta los 30 días anteriores.

# Cargando el dataset.
series = read_csv(path2, header=0, index_col=0)
values = series.values

# transform the time series data into supervised learning
data = series_to_supervised(values, n_in=6)

# Evaluando.
mae, y, yhat = walk_forward_validation(data, 30)

# Graficando esperado vs pronosticado.
pyplot.plot(y, label='Esperado')
pyplot.plot(yhat, label='Pronosticado')
pyplot.title('Predicción Modelo XGBoost sin tuning')
pyplot.legend()
pyplot.show()

# Pronosticando próximo día tomando en cuenta los 10 días anteriores.
# Cargando el dataset
series = read_csv(path2, header=0, index_col=0)
values = series.values

# Transformando las series de tiempo en aprendizaje supervisado.
train = series_to_supervised(values, n_in=30)

# Separando entre input y outputs.
trainX, trainy = train[:, :-1], train[:, -1]

# Fit al modelo.
model = XGBRegressor(objective='reg:squarederror', gpu_id=-1, max_depth=2)

model.fit(trainX, trainy)

# Input para la predicción.
row = values[-30:].flatten()

# Un paso de predicción.
yhat = model.predict(asarray([row]))
print('Input: %s, Predicted with XGBoost sin tuning: %.3f' % (row, yhat[0]))


###############################################################################
# XGBoost 2 con tuning.
###############################################################################
# # Buscamos los mejores hiperparámetros dentro del listado params.
# xgb = XGBRegressor(objective='reg:squarederror', gpu_id=-1)
#
# params = {
#     'min_child_weight': [5, 10, 40, 60],
#     'gamma': [0.5, 1.5, 3, 6],
#     'subsample': [0.6, 0.8, 1],
#     'colsample_bytree': [0.6, 0.8, 1],
#     'max_depth': [10, 20, 30],
#     'learning_rate': [0.3, 0.5, 0.8, 1],
#     'objective': ['reg:squarederror'],
# }
#
# param_comb = 100
# folds = 2
# skf = StratifiedKFold(n_splits=folds, shuffle=True)
# random_search = RandomizedSearchCV(xgb, param_distributions=params,
#                                    n_iter=param_comb,
#                                    scoring='neg_mean_absolute_error',
#                                    n_jobs=4,
#                                    cv=skf.split(trainX, trainy),
#                                    verbose=3)
#
# random_search.fit(trainX, trainy)
#
# print(random_search.cv_results_)
# print('\n Best estimator:')
# print(random_search.best_estimator_)
# print(
#     '\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (
#     folds, param_comb))
# print(random_search.best_score_ * 2 - 1)
# print('\n Best hyperparameters:')
# print(random_search.best_params_)
# results = pd.DataFrame(random_search.cv_results_)


# Se crea el nuevo modelo con los hiperparámetros encontrados.
def xgboost_forecast(train, testX):
    # transformando lista en arreglo.
    train = asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                         colsample_bynode=1, colsample_bytree=1, gamma=0.5,
                         gpu_id=-1,
                         importance_type='gain', interaction_constraints='',
                         learning_rate=0.3, max_delta_step=0, max_depth=10,
                         min_child_weight=40,
                         monotone_constraints='()',
                         n_estimators=100, n_jobs=16, num_parallel_tree=1,
                         random_state=0,
                         reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                         subsample=1,
                         tree_method='exact', validate_parameters=1,
                         verbosity=None, objective='reg:squarederror',
                         eval_metric='mae')
    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict(asarray([testX]))
    return yhat[0]


# Predicción.

model2 = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                      colsample_bynode=1, colsample_bytree=1, gamma=0.5,
                      gpu_id=-1,
                      importance_type='gain', interaction_constraints='',
                      learning_rate=0.3, max_delta_step=0, max_depth=10,
                      min_child_weight=40,
                      monotone_constraints='()',
                      n_estimators=100, n_jobs=16, num_parallel_tree=1,
                      random_state=0,
                      reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                      subsample=1,
                      tree_method='exact', validate_parameters=1,
                      verbosity=None, objective='reg:squarederror',
                      eval_metric='mae')

model2.fit(trainX, trainy)
series2 = read_csv(path2, header=0, index_col=0)
values2 = series2.values
data = series_to_supervised(values2, n_in=6)
mae2, y2, yhat2 = walk_forward_validation(data, 30)

# plot expected vs predicted
pyplot.plot(y2, label='Esperado')
pyplot.plot(yhat2, label='Pronosticado')
pyplot.title('Predicción Modelo XGBoost tuneado')
pyplot.legend()
pyplot.show()

# Pronosticando próximo día tomando en cuenta los 30 días anteriores y el mejor
# modelo encontrado.

# Input para la nueva predicción.
row = values2[-30:].flatten()

# Un paso de predicción.
yhat = model2.predict(asarray([row]))
print('Input: %s, Predicted with XGBoost con tuning: %.3f' % (row, yhat[0]))

# Resultados finales.
print('MAE de Prophet Default: %.3f' % df_p['mae'])
print('MAE de Prophet Hiper: %.3f' % df_p2['mae'])
print('MAE de XGBoost Default: %.3f' % mae)
print('MAE de XGBoost Hiper: %.3f' % mae2)
