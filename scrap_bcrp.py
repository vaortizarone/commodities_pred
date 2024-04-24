import requests
import pandas as pd
#%%
url_commodities = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api/PD04702XD-PD04704XD-PD04703XD-PD04701XD-PD04705XD/json/2000-01-03/2024-04-20"

commodities = pd.DataFrame(requests.get(url_commodities).json()['periods'])

commos = pd.DataFrame(commodities['values'].to_list(), columns=['Cobre', 'Plata', 'Zinc', 'Oro', 'Petroleo'])

commos.insert(0, 'Fecha', commodities['name'])

#%%
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
#%%

commos['Fecha'] = pd.to_datetime(commos['Fecha'].replace(meses_espanol_a_ingles, regex=True), format='%d.%b.%y').dt.date
#%%
url_macro_nac = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api/PD12301MD-PD04637PD/json/2000-01-03/2024-04-20"

tc_tint_nac = pd.DataFrame(requests.get(url_macro_nac).json()['periods'])
tc_tint = pd.DataFrame(tc_tint_nac['values'].to_list(), columns=['Tasa de Interes' , 'Tipo de Cambio'])
tc_tint.insert(0, 'Fecha', tc_tint_nac['name'])
tc_tint['Fecha'] = pd.to_datetime(tc_tint['Fecha'].replace(meses_espanol_a_ingles, regex=True), format='%d.%b.%y').dt.date
tc_tint = tc_tint.loc[959:,]

#%%
url_macro_nac_ipc = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api/PN01273PM/json/2000-01-03/2024-04-20"

ipc = pd.DataFrame(requests.get(url_macro_nac_ipc).json()['periods'])
ipc_peru = pd.DataFrame(ipc['values'].to_list(), columns=['IPC'])
ipc_peru.insert(0, 'Fecha', ipc['name'])
ipc_peru['Fecha'] = pd.to_datetime(ipc_peru['Fecha'].replace(meses_espanol_a_ingles, regex=True), format='%b.%Y').dt.date
ipc_peru.set_index('Fecha', inplace=True)
ipc_peru = ipc_peru.asfreq('D', method='ffill')
ipc_peru.reset_index(inplace=True)
ipc_peru['Fecha'] = pd.to_datetime(ipc_peru['Fecha']).dt.date

#%%

url_china = 'https://api.stlouisfed.org/fred/series/observations?series_id=DEXCHUS&api_key=c21166a7e51faba35bf9fa45c1a261c0&file_type=json'
usd_cny = pd.DataFrame(requests.get(url_china).json()['observations']).iloc[:, [2, 3]]
usd_cny.columns = ['Fecha' , 'USD-CNY']
usd_cny['Fecha']=pd.to_datetime(usd_cny['Fecha']).dt.date

url_aus = 'https://api.stlouisfed.org/fred/series/observations?series_id=DEXUSAL&api_key=c21166a7e51faba35bf9fa45c1a261c0&file_type=json'
usd_aus = pd.DataFrame(requests.get(url_aus).json()['observations']).iloc[:, [2, 3]]
usd_aus.columns = ['Fecha' , 'USD-AUS']
usd_aus['Fecha']=pd.to_datetime(usd_cny['Fecha']).dt.date

#%%

forecast_copper = pd.merge(tc_tint, ipc_peru, on='Fecha', how='inner') \
    .merge(ipc_peru, on='Fecha', how='inner') \
    .merge(commos, on='Fecha', how='inner') \
    .merge(usd_aus, on='Fecha', how='inner')\
    .merge(usd_cny, on='Fecha', how='inner')

forecast_copper = forecast_copper.iloc[:,[0,1,2,3,5,6,7,8,9,10,11]]
forecast_copper.rename(columns={'IPC_x': 'IPC'}, inplace=True)