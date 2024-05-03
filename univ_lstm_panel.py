import pandas as pd
import numpy as np
import requests
import datetime
import panel as pn
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
#%%

# Obtener la fecha actual
fecha_actual = datetime.date.today()

# Formatear la fecha en el formato YYYY-MM-DD
fecha_formateada = fecha_actual.strftime("%Y-%m-%d")
#%%
url_cobre = f"https://estadisticas.bcrp.gob.pe/estadisticas/series/api/PD04701XD/json/1970-01-03/{fecha_formateada}"

cob = pd.DataFrame(requests.get(url_cobre).json()['periods'])


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
cob['name'] = pd.to_datetime(cob['name'].replace(meses_espanol_a_ingles, regex=True), format='%d.%b.%y').dt.date
#%%

cob['values'] = pd.DataFrame(cob['values'].to_list())
#%%
cob['values'] = cob['values'].replace('n.d.', np.nan)
cob['values'] = pd.to_numeric(cob['values'])
cob['values'] = cob['values'].interpolate(method='linear')

cob = cob.set_index('name')


dataset = cob.values
scaler = MinMaxScaler(feature_range = (0, 1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:int(cob['values'].shape[0]*0.95), :]

x_train = []
y_train = []
#%%
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train, "\n")
        print(y_train)
#%%
x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(LSTM(128, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer = "adam", loss = "mean_squared_error")

model.fit(x_train, y_train, batch_size = 1, epochs = 1)


test_data = scaled_data[int(cob['values'].shape[0]*0.95) - 60: , :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[int(cob['values'].shape[0]*0.95):, :]
#%%
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

train = cob[:int(cob['values'].shape[0]*0.95)]
valid = cob[int(cob['values'].shape[0]*0.95):]
valid['Predictions'] = predictions


x_prediccion = np.array(test_data[-60:])
x_prediccion = np.reshape(x_prediccion, (1, x_prediccion.shape[0], 1))

cobre_tmrr = scaler.inverse_transform(model.predict(x_prediccion))

#%%


#%%
import plotly.express as px
from shared import app_dir, tips
from shiny import reactive, render
from shiny.express import input, ui
from shinywidgets import render_plotly
import faicons as fa

bill_rng = (min(tips.total_bill), max(tips.total_bill))

# Add page title and sidebar
ui.page_opts(title="Predicción del precio del cobre", fillable=True)

ICONS = {
    "user": fa.icon_svg("user", "regular"),
    "wallet": fa.icon_svg("wallet"),
    "currency-dollar": fa.icon_svg("dollar-sign"),
    "ellipsis": fa.icon_svg("ellipsis"),
}

with ui.layout_columns(fill=False):
    with ui.value_box(showcase=ICONS["user"]):
        "Total tippers"

        @render.express
        def total_tippers():
            tips_data().shape[0]

    with ui.value_box(showcase=ICONS["wallet"]):
        "Average tip"

        @render.express
        def average_tip():
            d = tips_data()
            if d.shape[0] > 0:
                perc = d.tip / d.total_bill
                f"{perc.mean():.1%}"

    with ui.value_box(showcase=ICONS["currency-dollar"]):
        "Average bill"

        @render.express
        def average_bill():
            d = tips_data()
            if d.shape[0] > 0:
                bill = d.total_bill.mean()
                f"${bill:.2f}"
