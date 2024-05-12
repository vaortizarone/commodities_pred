import pandas as pd
import numpy as np
import requests
import datetime
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
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime

# Supongamos que tienes tu DataFrame df_cob con las fechas como índice y los precios como valores
# También supondremos que cobre_tmrr es un objeto numpy.ndarray con el precio estimado del cobre para mañana

# Ejemplo de DataFrame de precios de cobre
fechas = pd.date_range(start='2024-01-01', end='2024-05-11')
precios = np.random.rand(len(fechas)) * 100  # Generar precios aleatorios para el ejemplo
df_cob = pd.DataFrame({'Precio': precios}, index=fechas)

# Precio estimado del cobre para mañana
cobre_tmrr = np.array([[448.80356]])

# Configurar la página Streamlit
st.title('Precio del Cobre')

# Gráfico de línea con precios de cobre a lo largo del tiempo
st.plotly_chart(
    go.Figure(
        data=[
            go.Scatter(
                x=df_cob.index,
                y=df_cob['Precio'],
                mode='lines',
                marker=dict(color='blue'),
                name='Precio de Cobre'
            )
        ],
        layout=go.Layout(
            title='Precio del Cobre a lo largo del tiempo',
            xaxis={'title': 'Fecha'},
            yaxis={'title': 'Precio (centavos de dólar por libra)'}
        )
    )
)

# Value box con el precio estimado del cobre para mañana
st.subheader("Precio estimado del cobre para mañana")
st.write(f"${cobre_tmrr[0][0]:.2f} centavos de dólar por libra")

# Fecha de actualización
st.subheader(f"Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")