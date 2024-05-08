import numpy as np 
import Utilities as util
import Externals as extt

yfd = extt.GetYFD()
(metrics, keys) = extt.GetLiveMetrics(yfd)
moves = extt.GetMoves(metrics, keys)
orderList = extt.CreateOrderList(moves, yfd)
cumOrders = {}
for sym in keys:
    cumOrders[sym] = 0

(eqt, pos) = GetAccountInfo()
for order in orderList:
    if order["side"] == "buy":
        cumOrders[order["sym"]] += order["qty"]
    else:
        available = 0
        for p in pos:
            if p.symbol == pos["sym"]:
                available = float(p.qty)
            
        cumOrders[order["sym"]] -= order["qty"]
        

for sym in cumOrders:
    if abs(cumOrders[sym] > 0.01):
        if cumOrders[sym] < 0:
            