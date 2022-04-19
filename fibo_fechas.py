# Python Module fibo_fechas

import time, os
from datetime import datetime, timedelta

def f_format(fecha):
    t = datetime.strptime(fecha, '%d %B, %Y %H:%M:%S')
    UTC1 = t - timedelta(hours=5)
    UTC2 = t - timedelta(hours=1)
    UTC1_3 = UTC1 + timedelta(hours=3)
    UTC2_3 = UTC2 + timedelta(hours=3)
    print('orig',str(UTC1_3),str(UTC2_3))
    return str(UTC1),str(UTC2)

def f_indicators(fecha):
    t = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
    UTC = t - timedelta(days=14)
    UTC1_3 = UTC - timedelta(hours=3)
    UTC2_3 = t - timedelta(hours=3)
    print('RSI/',str(UTC1_3), str(UTC2_3))
    return str(UTC)

def f_rev(fecha):
    t = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
    UTC = t
    UTC1 = t - timedelta(hours=4)
    UTC_1 = UTC + timedelta(hours=3)
    UTC1_1 = UTC1 + timedelta(hours=3)
    print('REV/',str(UTC1_1), str(UTC_1))
    return str(UTC1), str(UTC)


