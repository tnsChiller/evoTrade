import Externals as extt
import Utilities as util
pSets = util.LoadFile("pSets")

print("\nStarting ...")
go = True
while True:
    clock = extt.GetClock()
    if clock.is_open:
        if util.GetGoTime(31, 0):
        # if True:
            if go:
                yfd = extt.GetYFD()
                (metrics, keys) = util.GetLiveMetrics(yfd)
                moves = util.GetMoves(pSets, metrics, keys)
                orderList = util.CreateOrderList(moves, yfd, pSets)
                extt.ExecuteOrders(orderList, keys)
                go = False
                
        else:
            go = True
            
    else:
        print("Market is closed.")
        print(f"Time: {extt.GetTime()}")
    
    # util.RunSummary()
    extt.RunPositionSummary()
    for cycle in range(12):
        util.WaitCycle()
        