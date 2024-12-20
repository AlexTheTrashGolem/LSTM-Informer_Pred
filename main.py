from module import *


# https://rapidapi.com/apidojo/api/yahoo-finance1
# url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/stock/v3/get-historical-data"

# symbol = sys.argv[1] if len(sys.argv) > 1 else "BTC-USD"
# querystring = {"symbol":symbol,"region":"US"}
# headers = {
# 	"X-RapidAPI-Key": "6b71a5a904mshab121136b9a463ep185f45jsn1a23158a760b",
# 	"X-RapidAPI-Host": "apidojo-yahoo-finance-v1.p.rapidapi.com"
# }

# response = requests.get(url, headers=headers, params=querystring).json()
#
# if not response:
#     raise Exception('Symbol does not exist.')
#


response = fetch_historical_data()
response_df, normalized_data, std, mean = parse_yahoo_response(response)
exp = load_model_informer()
# response.json()['values']


# with open('model.pkl', 'rb') as f:
#         model = pickle.load(f)
# model=pickle.load(open('model.pkl','rb'))
model = load_model_lstm()

lstm_predictions = eval_model_lstm(model, response_df)

inf_middle = eval_model_inf(exp, normalized_data, std, mean)

preds = eval_model(lstm_predictions, inf_middle, response_df)

middle = combined_prediction(preds)


env = TradingEnv(balance_amount=100, balance_unit='USDT', trading_fee_multiplier=0.99925)
middle = ema_ribbon(middle)
print(middle)
for i in range(len(middle)):
    if env.balance_unit == 'USDT':
        if middle['Signal'].iloc[i] == 'buy':
            env.buy(response_df['close'].iloc[i], middle.index[i])
            # print(middle.index[i].date(), "Покупка")
        # else:
            # print(middle.index[i].date(), "Утримання")

    if env.balance_unit != 'USDT':
        if middle['Signal'].iloc[i] == 'sell':  # sell signal
            env.sell(response_df['close'].iloc[i], middle.index[i])
            # print(middle.index[i].date(), "Продаж")
        # else:
            # print(middle.index[i].date(), "Утримання")

if env.balance_unit != 'USDT':
    env.sell(middle['close'].iloc[-1], middle .index[-1])
last_records = response_df.head(104).reset_index(drop=True)
middle = pd.concat([last_records, middle.reset_index(drop=True)], axis=1)

print(f'num buys: {len(env.buys)}')
print(f'num sells: {len(env.sells)}')
print(f'ending balance: {env.balance_amount} {env.balance_unit}')