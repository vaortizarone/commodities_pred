import pandas as pd
import numpy as np
import requests
import xgboost as xgb
import shap
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

def obtener_predicciones_cobre():
    # Definir URL para obtener datos de commodities
    url_commodities = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api/PD04702XD-PD04704XD-PD04703XD-PD04701XD-PD04705XD/json/2000-01-03/2024-04-20"

    # Obtener datos de commodities
    commodities = pd.DataFrame(requests.get(url_commodities).json()['periods'])
    commos = pd.DataFrame(commodities['values'].to_list(), columns=['Cobre', 'Plata', 'Zinc', 'Oro', 'Petroleo'])
    commos.insert(0, 'Fecha', commodities['name'])

    # Mapear nombres de meses de español a inglés
    meses_espanol_a_ingles = {
        'Ene': 'Jan',
        'Feb': 'Feb',
        'Mar': 'Mar',
        'Abr': 'Apr',
        'May': 'May',
        'Jun': 'Jun',
        'Jul': 'Jul',
        'Ago': 'Aug',
        'Set': 'Sep',
        'Oct': 'Oct',
        'Nov': 'Nov',
        'Dic': 'Dec'
    }
    commos['Fecha'] = pd.to_datetime(commos['Fecha'].replace(meses_espanol_a_ingles, regex=True), format='%d.%b.%y').dt.date

    # Definir URL para obtener datos macroeconómicos
    url_macro_nac = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api/PD12301MD-PD04637PD/json/2000-01-03/2024-04-20"

    # Obtener datos macroeconómicos
    tc_tint_nac = pd.DataFrame(requests.get(url_macro_nac).json()['periods'])
    tc_tint = pd.DataFrame(tc_tint_nac['values'].to_list(), columns=['Tasa de Interes' , 'Tipo de Cambio'])
    tc_tint.insert(0, 'Fecha', tc_tint_nac['name'])
    tc_tint['Fecha'] = pd.to_datetime(tc_tint['Fecha'].replace(meses_espanol_a_ingles, regex=True), format='%d.%b.%y').dt.date
    tc_tint = tc_tint.loc[959:,]

    # Obtener datos macroeconómicos de IPC
    url_macro_nac_ipc = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api/PN01273PM/json/2000-01-03/2024-04-20"
    ipc = pd.DataFrame(requests.get(url_macro_nac_ipc).json()['periods'])
    ipc_peru = pd.DataFrame(ipc['values'].to_list(), columns=['IPC'])
    ipc_peru.insert(0, 'Fecha', ipc['name'])
    ipc_peru['Fecha'] = pd.to_datetime(ipc_peru['Fecha'].replace(meses_espanol_a_ingles, regex=True), format='%b.%Y').dt.date
    ipc_peru.set_index('Fecha', inplace=True)
    ipc_peru = ipc_peru.asfreq('D', method='ffill')
    ipc_peru.reset_index(inplace=True)
    ipc_peru['Fecha'] = pd.to_datetime(ipc_peru['Fecha']).dt.date

    # Obtener datos de tipo de cambio USD/CNY
    url_china = 'https://api.stlouisfed.org/fred/series/observations?series_id=DEXCHUS&api_key=c21166a7e51faba35bf9fa45c1a261c0&file_type=json'
    usd_cny = pd.DataFrame(requests.get(url_china).json()['observations']).iloc[:, [2, 3]]
    usd_cny.columns = ['Fecha' , 'USD-CNY']
    usd_cny['Fecha'] = pd.to_datetime(usd_cny['Fecha']).dt.date

    # Obtener datos de tipo de cambio USD/AUS
    url_aus = 'https://api.stlouisfed.org/fred/series/observations?series_id=DEXUSAL&api_key=c21166a7e51faba35bf9fa45c1a261c0&file_type=json'
    usd_aus = pd.DataFrame(requests.get(url_aus).json()['observations']).iloc[:, [2, 3]]
    usd_aus.columns = ['Fecha' , 'USD-AUS']
    usd_aus['Fecha'] = pd.to_datetime(usd_cny['Fecha']).dt.date

    # Combinar datos macroeconómicos y de commodities
    forecast_copper = pd.merge(tc_tint, ipc_peru, on='Fecha', how='inner') \
        .merge(ipc_peru, on='Fecha', how='inner') \
        .merge(commos, on='Fecha', how='inner') \
        .merge(usd_aus, on='Fecha', how='inner')\
        .merge(usd_cny, on='Fecha', how='inner')

    # Seleccionar columnas relevantes y renombrar
    forecast_copper = forecast_copper.iloc[:,[0,1,2,3,5,6,7,8,9,10,11]]
    forecast_copper.rename(columns={'IPC_x': 'IPC'}, inplace=True)

    # Convertir columnas a numéricas
    forecast_copper.iloc[:, 1:] = forecast_copper.iloc[:, 1:].apply(lambda x: pd.to_numeric(x, errors='coerce'))

    # Crear columna de 'Cobre Dia'
    forecast_copper['Cobre'].shift(-1)
    day_copper = pd.DataFrame(forecast_copper)
    day_copper['Cobre Dia'] = day_copper['Cobre'].shift(-1)
    day_copper = day_copper.dropna()

    # Preparar datos para entrenamiento
    day_copper.set_index('Fecha', inplace=True)
    day_copper = day_copper.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    n_train = round(len(day_copper) * 0.8)
    train = day_copper.iloc[:int(n_train)]
    test = day_copper.iloc[int(n_train):]
    X_train = train.drop(columns=['Cobre Dia'])
    y_train = train['Cobre Dia']
    X_test = test.drop(columns=['Cobre Dia'])
    y_test = test['Cobre Dia']

    # Tuning de hiperparámetros con GridSearch (comentado para no ejecutar por defecto)
    '''
    param_grid = {
        'max_depth': [3, 6, 9, 12],
        'learning_rate': [0.01, 0.1, 0.2, 0.3, 0.4],
        'subsample': [0.4, 0.6, 0.8, 1],
        'gamma': [0, 0.1, 0.2, 0.4]
    }
    grid_search = GridSearchCV(estimator=xgb.XGBRegressor(objective='reg:squarederror'),
                               param_grid=param_grid,
                               scoring='neg_mean_squared_error',
                               cv=3,
                               n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    '''

    # Entrenamiento del modelo XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    params = {'objective': 'reg:squarederror', 'max_depth': 12, 'learning_rate': 0.4, 'gamma': 0.2, 'subsample': 0.4}
    model_xgb = xgb.train(params, dtrain)
    predictions_xgb = model_xgb.predict(dtest)
    mse_xgb = mean_squared_error(y_test, predictions_xgb)
    rmse_xgb = np.sqrt(mse_xgb)
    mae_xgb = mean_absolute_error(y_test, predictions_xgb)
    model_xgb.save_model("xgb_cobre.json")

    # Feature Importance
    explainer = shap.TreeExplainer(model_xgb)
    shap_values = explainer(X_test)

    return shap_values, "xgb_cobre.json" , rmse_xgb, mae_xgb
#%%


