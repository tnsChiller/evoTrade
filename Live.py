import Externals as extt
import Utilities as util

go = True
while True:
    clock = extt.GetClock()
    if clock.is_open:
        if util.GetGoTime(30, 30):
            if go:
                yfd = extt.GetYFD()
                (metrics, keys) = extt.GetLiveMetrics(yfd)
                moves = extt.GetMoves(metrics, keys)
                orderList = extt.CreateOrderList(moves, yfd)
                extt.ExecuteOrders(orderList, keys)
                go = False
                
        else:
            go = True
    
    # util.RunSummary()
    util.WaitCycle()