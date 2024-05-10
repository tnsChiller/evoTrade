import Utilities as util

obl = util.LoadFile("orderBigList")
sort = {}
for order in obl:
    if obl[order][model] not in sort:
        sort[obl[order][model]] = {}