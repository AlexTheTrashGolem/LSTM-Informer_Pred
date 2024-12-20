from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import numpy as np
# import plotly.express as px
import pandas as pd
import pickle
import keras
import torch
import sys
import os
import datetime
if not 'Informer2020' in sys.path:
    sys.path += ['Informer2020']
from utils.tools import dotdict
from exp.exp_informer import Exp_Informer
import requests
from key import headers

headers = {
	"X-RapidAPI-Key": "6b71a5a904mshab121136b9a463ep185f45jsn1a23158a760b",
	"X-RapidAPI-Host": "apidojo-yahoo-finance-v1.p.rapidapi.com"
}



Exp = Exp_Informer

args = dotdict()

args.model = 'informer' # model of experiment, options: [informer, informerstack, informerlight(TBD)]

args.data = 'custom' # data
args.root_path = './data/' # root path of data file
args.data_path = 'Temp.csv' # data file
args.features = 'MS' # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
args.target = 'close' # target feature in S or MS task
args.freq = 'h' # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
args.checkpoints = './informer_checkpoints' # location of model checkpoints

args.seq_len = 96 # input sequence length of Informer encoder
args.label_len = 48 # start token length of Informer decoder
args.pred_len = 12 # prediction sequence length
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

args.enc_in = 4 # encoder input size
args.dec_in = 4 # decoder input size
args.c_out = 1 # output size
args.factor = 5 # probsparse attn factor
args.d_model = 512 # dimension of model
args.n_heads = 8 # num of heads
args.e_layers = 2 # num of encoder layers
args.d_layers = 1 # num of decoder layers
args.d_ff = 2048 # dimension of fcn in model
args.dropout = 0.05 # dropout
args.attn = 'prob' # attention used in encoder, options:[prob, full]
args.embed = 'timeF' # time features encoding, options:[timeF, fixed, learned]
args.activation = 'gelu' # activation
args.distil = True # whether to use distilling in encoder
args.output_attention = False # whether to output attention in ecoder
args.mix = True
args.padding = 0
args.freq = 'h'

args.batch_size = 32 
args.learning_rate = 0.0001
args.loss = 'mse'
args.lradj = 'type1'
args.use_amp = False # whether to use automatic mixed precision training

args.num_workers = 0
args.itr = 1
args.train_epochs = 6
args.patience = 3
args.des = 'exp'

args.use_gpu = True if torch.cuda.is_available() else False
args.gpu = 0
args.inverse = True
args.use_multi_gpu = False
args.devices = '0,1,2,3'

args.inverse = True

args.detail_freq = args.freq
args.freq = args.freq[-1:]

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

aim = 'close'

window = 5

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]
checkpoint_path = 'informer_checkpoint.pth'

def fetch_historical_data():
    url = "https://api.twelvedata.com/time_series?symbol=BTC/USD&interval=1h&outputsize=200&apikey=1e883412b62940ccb48e5573b9b6a6f5"
    response = requests.get(url, headers=headers)
    return response


def parse_yahoo_response(response):
    response_df = pd.DataFrame(response.json()['values'])
    response_df = response_df.rename(columns={'datetime':'date'}, errors="raise")

    response_df = response_df.iloc[::-1].reset_index(drop=True)

    columns = ['open','high','low','close']
    for i in columns:
        response_df[i] = response_df[i].astype(float)
    
    std, mean = response_df.std(axis=0), response_df.mean(axis=0)
    normalized_data = pd.DataFrame()
    normalized_data['date'] = response_df['date']
    normalized_data[columns] = (response_df[columns] - mean) / std    
    
    return response_df, normalized_data, std, mean 

def load_model_informer():
    exp = Exp(args)
    exp.model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    return exp


def load_model_lstm():
    return load_model('model.h5')
    
    
def eval_model_lstm(model, data):
    lstm_response_df = pd.DataFrame()
    # lstm_response_df = lstm_response_df[['open','high','low','close','volume']]

    # lstm_response_df['adj_close'] = lstm_response_df['close']
    lstm_response_df['close'] = data['close']

    print(lstm_response_df.tail())
    l = len(data)
    lstm_predictions = []
    for i in range(window*4,l):
        lstm_predictions += [lstm_middle_band(lstm_response_df[i - window*4 : l], model, window, 'close')]
    lstm_predictions = pd.DataFrame(lstm_predictions)
    return lstm_predictions
    
    
    
    
def eval_model_inf(model, data, std, mean):
    Window = 42
    inf_middle = [0]*(len(data)-96)
    model.args.data_path = 'Temp.csv'
    if not os.path.exists('data'):
        os.makedirs('data')
    # columns = ['open','high','low','close','volume', 'volume_btc']
    # response_df['date'] = np.nan
    for i in range(96,len(data)):
        temp_df = pd.DataFrame(data[i-96:i])
        temp_df.to_csv('./data/Temp'+str(i)+'.csv', index=False)
    for i in range(96,len(data)):
        args.data_path = 'Temp'+str(i)+'.csv'
        model = Exp(args)
        model.model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
        inf_middle[i-96] = middle_band(model)
    inf_middle = ((pd.DataFrame(inf_middle) * std[3]) + mean[3])[11]
    return inf_middle

    
    
    
def plot_results(plot_df, buys, sells):
    fig = go.Figure(data=[go.Candlestick(x=plot_df.index,
                open=plot_df['open'],
                high=plot_df['high'],
                low=plot_df['low'],
                close=plot_df['close'])])

    fig.update_layout(xaxis_rangeslider_visible=False)

    fig.add_trace(go.Scatter(x=plot_df.index, 
                             y=plot_df['middle_band'],
                             type='scatter', 
                             mode='lines',
                             line=dict(color='grey'),
                             name='Середня лiнiя'))

    fig.add_trace(go.Scatter(x=plot_df.index, 
                             y=plot_df['EMA_5'],
                             type='scatter', 
                             mode='lines',
                             line=dict(color='lightgrey'),
                             name='EMA_5'))

    fig.add_trace(go.Scatter(x=plot_df.index, 
                             y=plot_df['EMA_5'],
                             type='scatter', 
                             mode='lines',
                             line=dict(color='lightgrey'),
                             name='EMA_5'))

    fig.add_trace(go.Scatter(x=plot_df.index,
                             y=plot_df['EMA_15'],
                             type='scatter',
                             mode='lines',
                             line=dict(color='lightgrey'),
                             name='EMA_15'))

    fig.add_trace(go.Scatter(x=[buy[1] for buy in buys if buy[0] == 'BTC'], 
                             y=[buy[2] for buy in buys if buy[0] == 'BTC'],
                             type='scatter', 
                             mode='markers',
                             marker=dict(symbol='x',color='blue', size=10),
                             name='Покупки'))

    fig.add_trace(go.Scatter(x=[sell[1] for sell in sells if sell[0] == 'BTC'], 
                             y=[sell[2] for sell in sells if sell[0] == 'BTC'],
                             type='scatter', 
                             mode='markers',
                             marker=dict(symbol='x',color='orange', size=10),
                             name='Продажі'))

    fig.update_xaxes(range = [plot_df.index[0],plot_df.index[-1]])
    fig.update_yaxes(range = [min(plot_df['low'])*.99,max(plot_df['high'])*1.01])

    return fig.show()


class TradingEnv:
    def __init__(self, balance_amount, balance_unit, trading_fee_multiplier):
        self.balance_amount = balance_amount
        self.balance_unit = balance_unit
        self.buys = []
        self.sells = []
        self.trading_fee_multiplier = trading_fee_multiplier
        
    def buy(self, buy_price, time):
        self.balance_amount = (self.balance_amount / buy_price) / self.trading_fee_multiplier
        self.balance_unit = 'BTC'
        self.buys.append(['BTC', time, buy_price])
        
    def sell(self, sell_price, time):
        self.balance_amount = self.balance_amount * sell_price / self.trading_fee_multiplier
        self.sells.append( [self.balance_unit, time, sell_price] )
        self.balance_unit = 'USDT'




def normalise_zero_base(continuous):
    return continuous / continuous.iloc[0] - 1


def extract_data(continuous, window_len=5, zero_base=True):
    window_data = []
    for idx in range(len(continuous) - window_len):
        tmp = continuous[idx: (idx + window_len)].copy()
        if zero_base:
            tmp = normalise_zero_base(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)



def predict(exp, load=False):
    pred_data, pred_loader = exp._get_data(flag='pred')
#     print(pred_data.df_raw.shape)

    exp.model.eval()

    preds = []

    for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
        batch_x = batch_x.float().to(exp.device)
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float().to(exp.device)
        batch_y_mark = batch_y_mark.float().to(exp.device)

        # decoder input

        dec_inp = torch.zeros([batch_y.shape[0], exp.args.pred_len, batch_y.shape[-1]]).float()

        dec_inp = torch.cat([batch_y[:,:exp.args.label_len,:], dec_inp], dim=1).float().to(exp.device)
        # encoder - decoder

        outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        
        f_dim = -1 if exp.args.features=='MS' else 0
        batch_y = batch_y[:,-exp.args.pred_len:,f_dim:].to(exp.device)

        pred = outputs.detach().cpu().numpy()#.squeeze()

        preds.append(pred)

    preds = np.array(preds)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])


    return preds

def calculate_rmse(df, col1, col2):
    squared_diff = (df[col1] - df[col2]) ** 2
    mean_squared_diff = squared_diff.mean()
    rmse = np.sqrt(mean_squared_diff)
    return rmse

def lstm_middle_band(data, model, window_len=5, aim='close'):
    extracted_data = extract_data(data.iloc[-window_len*2:], window_len)
    predicted_changes = model.predict(extracted_data).squeeze()
    predicted_changes = np.minimum(predicted_changes, 0.3)
    predicted_changes = np.maximum(predicted_changes, -0.3)
    predicted_values = [data[aim].iloc[-1]]
    for i in predicted_changes:
        predicted_values+=[predicted_values[-1]*(1+i)]
    middle_band = (data[aim].iloc[-1] + data[aim].values.mean() + (sum(predicted_values)/len(predicted_values)))/3
    return middle_band


def middle_band(exp, window_len=5, aim='close'):
#     extracted_data = extract_data(data.iloc[-window_len*2:], window_len)
#     predicted_changes = model.predict(extracted_data).squeeze()
    predicted_changes = predict(exp).squeeze()
    
    # print(predicted_changes.shape)
#     predicted_changes = np.minimum(predicted_changes, 0.3)
#     predicted_changes = np.maximum(predicted_changes, -0.3)
    
    
    middle_band = predicted_changes
#     predicted_values = [data[aim].iloc[-1]]
#     for i in predicted_changes:
#         predicted_values+=[predicted_values[-1]*(1+i)]
#     middle_band = (data[aim].iloc[-1] + data[aim].values.mean() + (sum(predicted_values)/len(predicted_values)))/3
    return middle_band


# def bollinger_band(data, middle_band, window, nstd):
#     std = data.rolling(window = window).std()
#     upper_band = middle_band + std * nstd
#     lower_band = middle_band - std * nstd
#     return upper_band, lower_band


def ema_ribbon(data, ema_windows=[5, 10, 15, 20, 25]):
    # Calculate EMA ribbons
    ema_df = pd.DataFrame()
    for window in ema_windows:
        ema_df[f"EMA_{window}"] = data.ewm(span=window, adjust=False).mean()

    # Initial signal logic based on EMA relationships
    ema_df['RawSignal'] = 'hold'
    ema_df['RawSignal'] = pd.np.where(
        ema_df[f"EMA_{ema_windows[0]}"] > ema_df[f"EMA_{ema_windows[-1]}"], "buy",
        pd.np.where(
            ema_df[f"EMA_{ema_windows[0]}"] < ema_df[f"EMA_{ema_windows[-1]}"], "sell",
            "hold"
        )
    )

    # Refine signals with state tracking
    current_state = "hold"
    refined_signals = []

    for signal in ema_df['RawSignal']:
        if signal == "buy" and current_state != "buy":
            refined_signals.append("buy")
            current_state = "buy"
        elif signal == "sell" and current_state != "sell":
            refined_signals.append("sell")
            current_state = "sell"
        else:
            refined_signals.append("hold")

    ema_df['Signal'] = refined_signals
    ema_df['middle_band'] = data
    return ema_df


def eval_model(lstm_predictions, inf_middle, response_df):
    lstm_records = lstm_predictions.tail(104).reset_index(drop=True)
    preds = pd.concat([lstm_records, inf_middle.reset_index(drop=True)], axis=1)
    last_records = response_df['close'].tail(104).reset_index(drop=True)
    preds = pd.concat([last_records, preds.reset_index(drop=True)], axis=1)
    return preds

def calculate_rmse(df, col1, col2):
    squared_diff = (df[col1] - df[col2]) ** 2
    mean_squared_diff = squared_diff.mean()
    rmse = np.sqrt(mean_squared_diff)
    return rmse


def model_weights(data):
    lstm_err = calculate_rmse(data, 'close', 0)
    inf_err = calculate_rmse(data, 'close', 11)
    lstm_weight = 1 - (lstm_err/(lstm_err+inf_err))
    inf_weight = 1 - lstm_weight
    return lstm_weight, inf_weight


def combined_prediction(data):
    lstm_weight, inf_weight = model_weights(data)
    middle_band = data[0] * lstm_weight + data[11] * inf_weight
    return middle_band




