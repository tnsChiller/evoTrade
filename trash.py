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

for order in orderList:
    if oder["side"] == "buy":
        cumOrders[order["sym"]] += qty
    else:
        cumOrders[order["sym"]] -= qty
        

(eqt, pos) = GetAccountInfo
for sym in cumOrders:
    if abs(cumOrders[sym] > 0.01):
        if cumOrders[sym] < 0:
            