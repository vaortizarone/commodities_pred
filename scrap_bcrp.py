import requests

url = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api/PD31889XD-PD31888XD-PD31887XD-PD04705XD-PD04702XD-PD04704XD-PD04703XD-PD04701XD/json/2000-01-03/2024-04-20"

response = requests.get(url)

data = response.json()

data