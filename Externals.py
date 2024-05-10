import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from Constants import playList
import datetime
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)
key = 'PKKQMCMWMEYQ9S7VEBPN'
secret = 'xRhpoy3649uxkfhuSMTNbd4SHntRgPCaogd1izCa'
client = TradingClient(key, secret, paper=True)

def GetYFD():
    yfd = yf.download(playList, period ='30d', interval = '1h')
    
    return yfd

def GetAccountInfo():
    acc = client.get_account()
    eqt = float(acc.equity)
    pos = client.get_all_positions()
    
    return(eqt, pos)

def ExecuteOrders(orderList, keys):
    cumOrders = {}
    for sym in keys:
        cumOrders[sym] = 0
    
    for order in orderList:
        if order["side"] == "buy":
            cumOrders[order["sym"]] += order["qty"]
            
        else:
            cumOrders[order["sym"]] -= order["qty"]
            
    
    for sym in cumOrders:
        if abs(cumOrders[sym] > 0.01):
            if cumOrders[sym] < 0:
                side = OrderSide.SELL
            else:
                side = OrderSide.BUY
                order = MarketOrderRequest(symbol = sym,
                                           qty = cumOrders[sym],
                                           side = side,
                                           time_in_force = TimeInForce.DAY)
                order = client.submit_order(order_data = order)
                print(order)
                
def GetClock():
    
    return client.get_clock()

def GetTime():
    
    return datetime.datetime.now()