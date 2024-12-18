from module import *
import sys
import requests
from key import headers

# https://rapidapi.com/apidojo/api/yahoo-finance1
url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/stock/v3/get-historical-data"

symbol = sys.argv[1] if len(sys.argv) > 1 else "BTC-USD"

querystring = {"symbol":symbol,"region":"US"}

# headers = {
# 	"X-RapidAPI-Key": "6b71a5a904mshab121136b9a463ep185f45jsn1a23158a760b",
# 	"X-RapidAPI-Host": "apidojo-yahoo-finance-v1.p.rapidapi.com"
# }

response = requests.get(url, headers=headers, params=querystring).json()

if not response:
    raise Exception('Symbol does not exist.')

with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

response_df = pd.DataFrame([response['prices'][len(response['prices']) -1-i] for i in range(len(response['prices']))])
response_df = response_df[['open','high','low','close','adjclose','volume']]
l = len(response_df) 

working_prediction = middle_band(response_df[l - window*2 : l], model, window, 'close')



predicted_upper, predicted_lower = bollinger_band(response_df['close'], working_prediction, 20, 3)

signal = "Утримуватись"
if working_prediction > predicted_upper.iloc[-1]:
    signal = "Продавати"
if working_prediction < predicted_lower.iloc[-1]:
    signal = "Покупати"


print(signal)