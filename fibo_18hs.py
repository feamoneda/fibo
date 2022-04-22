SLACK_BOT_TOKEN="xoxb-2772745627024-2844989989255-0gEfon6PFK15kS1YIQk4HldE"
import statsmodels.api as sm
import seaborn as sns
import pandas as pd
from pandas import DataFrame
from collections import OrderedDict, Counter
import numpy as np
import time, os
from datetime import datetime, timedelta
import pytz
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
# plt.rcParams["figure.figsize"] = (15,7)
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import talib as ta
from binance.client import Client
import warnings
from decimal import Decimal, getcontext
import decimal
from sklearn.linear_model import LinearRegression
import math
from requests.exceptions import Timeout  # this handles ReadTimeout or ConnectTimeout
from sklearn.preprocessing import StandardScaler
from fractions import Fraction
from statistics import mean
from datetime import date
from filestack import Client as FS_Client
#from filestack import Security
import logging
#logging.basicConfig(level=logging.DEBUG)
from slack_sdk import WebClient, WebhookClient
#warnings.filterwarnings('ignore')
from PIL import Image, ImageFont, ImageDraw
from urllib.parse import urljoin
import argparse
import os
import sys
import glob
from pathlib import Path
#cookie_del = glob.glob("config/*cookie.json")
#os.remove(cookie_del[0])
#sys.path.append(os.path.join(sys.path[0], "../"))
from instagrapi import Client as Cl
from instagrapi.story import StoryBuilder
from instagrapi.types import StoryMention, StoryMedia, StoryLink, StoryHashtag
import tweepy
from slack_sdk.errors import SlackApiError
import requests
from requests.structures import CaseInsensitiveDict
import random
import fibo_fechas


tokenSlack = SLACK_BOT_TOKEN
clientSlack = WebClient(token=tokenSlack)

logger = logging.getLogger(__name__)

client = Client('fqnT2APdGB0hDJyJFaznx8hQHNFeRTfqwnIt4TfYxzBtsq4q9CqIB8tVySoDvlH8', 'VCIlmPwKTL0lzoSjxszbpB2MjMwYVWYQmf2QcL9JmZmo4GO8l5N3fB0gzQuUVKrl', {"verify": False, "timeout": 70})
data = client.get_all_tickers()

#data = ['SLPUSDT','ALGOUSDT','BLZUSDT','BRDBTC','ETHUSDT']

# Authenticate to Twitter
auth = tweepy.OAuthHandler("gZjXy1g6HZdxOWPZ6CsXI7Vdr", "ll5sP6Co7U2pcMySk2ScTCtmmh5KomPWRHifZfeEQ834HeG3IN")
auth.set_access_token("16885094-BqOvh8Pba0FLa2eUsNhMqliq7Q0tnh2At6eYGNvcO", "EMJ9kF9jZMieN0WLz1JUPb60tvzf8C0k0o6o96rIwiuaq")
api = tweepy.API(auth)

local_time1 = int(round(time.time() * 1000))
server_time = client.get_server_time()
diff1 = server_time['serverTime'] - local_time1

if( diff1 < 1000 ):
    print('En horario')
    print('Server:',datetime.fromtimestamp(round(server_time['serverTime'])/1000))
    # print(server_time['serverTime'])

df = []
df = pd.DataFrame(data, columns = ['symbol'])

def getFecha(t):
    fecha1 = datetime.strptime(t, '%d %B, %Y %H:%M:%S') #fechaUTC("31 May, 2021 06:00:00")
    fecha1 = fecha1 - timedelta(hours=3)
    fecha2 = fecha1 - timedelta(days=15, hours=6)

    f00 = fecha1 - timedelta(days=15, hours=3)
    f01 = fecha1 - timedelta(days=7, hours=3)
    f02 = fecha1 - timedelta(days=3, hours=3)
    f03 = fecha1 - timedelta(days=1, hours=3)
    f04 = fecha1 - timedelta(hours=3)
    f05 = fecha1 - timedelta(days=44, hours=3)
    f06 = fecha1 - timedelta(hours=12)
    return fecha1, fecha2, f00, f01, f02, f03, f04, f05, f06

def get_prices(ticker, interval, desde, hasta):
    klines = client.get_historical_klines(ticker, interval, desde, hasta)
    historico = pd.DataFrame(klines)
    # print(historico.info())
    if ( len(historico.columns) == 0 ):
        # print("No data")
        return
    else:
        historico = historico.rename(columns = {0: 'Date', 1: 'Open', 2 : 'High', 3 : 'Low', 4 : 'Close', 5: 'Volumen'}, inplace = False)
        historico = historico.iloc[:, [True, True, True, True, True, True, False, False, False, False,False, False]]
        #for line in klines:
            #line[0] = line[0].round(3)
            #line[0] = (line[0]*1000)+10800000
              #fecha = pd.to_datetime(line[0], unit='ms', format=None).to_pydatetime() + timedelta(hours=3)#%H:%M:%S
              #fecha = fecha.strftime("%Y-%d-%m %H:%M:%S")
            #historico.Date.replace({line[0]: fecha}, inplace=True)
        return historico

def fechaUTC(fecha):
    t = datetime.strptime(fecha, '%d %B, %Y %H:%M:%S')
    UTC = t + timedelta(hours=3)
    #print(str(UTC))
    return str(UTC)

def fechaFormato(fecha):
    e = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
    e = e + timedelta(hours=3)
    return e.strftime("%Y-%d-%m %H:%M:%S")

def get_coef(dataframe):
    # diasanteriores = [tiempo14.strftime('%Y-%d-%m %H:%M UTC')]
    coefs = []

    histr = dataframe
    if( histr is not None and histr.empty is not True):
            fecha = dataframe.iloc[:, 0]
            scaler = StandardScaler()
            e = scaler.fit_transform(pd.DataFrame(fecha))

            coeficiente = regressionPredict(e, dataframe.iloc[:, 1].values.reshape(-1, 1))
            # coeficiente = math.floor(coeficiente)
            coefs.append(coeficiente) # agrego al array todos los coefs

            if ( str(coeficiente) == '1'):
                m = 'Bueno'
            if ( str(coeficiente) == '2'):
                m = 'Bueno+'
            if ( str(coeficiente) == '3'):
                m = 'Muy bueno'
            else:
                m = 'No'

    return m, coefs

def regressionPredict(e1, e2):
        #REQ. DF SEGUN FECHAS
        vector = np.vectorize(np.float)
        X = e1 #dataframe.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
        Y = e2 #dataframe.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
        Y = vector(Y)
        linear_regressor = LinearRegression()  # create object for the class
        linear_regressor.fit(X, Y)  # perform linear regression
        Y_pred = linear_regressor.predict(X)  # make predictions
        a = linear_regressor.coef_ #[0][0] # coeficiente
        b = linear_regressor.intercept_
        c = getattr(a, "tolist", lambda: a)() # paso a float
        # d = str.format('{0:.2f}',a) # redondeo
        # j = str(d) # string porque no funciona como float para la comparacion
        return a

def getDelta(dataframe):
    histr = dataframe
    if( histr is not None and histr.empty is not True):
        price_min = Decimal(1000000)
        price_max = Decimal(0)
        for day in dataframe.itertuples(index=True, name='Pandas'):
            if Decimal(day.Close) < price_min:
                price_min = Decimal(day.Close)
            if price_max < Decimal(day.High):
                price_max = Decimal(day.High)
        delta = 100 * (price_max - price_min) / price_min
        return math.floor(delta)

def getDifference (dataframe) :
   #Valor mas alto y bajo de ese tiempo
    a = 0
    histr = dataframe
    if( histr is not None and histr.empty is not True):
        bajo = Decimal(10000)
        alto = Decimal(0)
        for dia in histr.itertuples(index=True, name='Pandas'):
            a += 1
            if Decimal(dia.Low) < bajo:
                bajo = Decimal(dia.Low)
                #GETFECHA
                fecha = dia.Date
                # CHEQUEAR print(a,'Fecha',fecha)
                if( dia.Date >= fecha ):
                    alto = Decimal(dia.High)
                    # CHEQUEAR print(a,'ALTO',dia.Date, alto)

    #print( str(Decimal(dia.Low))+' < '+str(bajo) )
    # print('B',str(bajo))
    # print('A',str(alto))
    if (alto != 0):
        difference = (100-((bajo*100)/alto))
    else:
        difference = 0
    return round(difference,2)

def difOk(x, change):
    if x>change:
        #print(x,change,str('Ok'))
        return str('Ok')
    else:
        #print(x,change,str('No'))
        return str('No')

def difOkrs(precio, rs, v1, v2):
    if rs == 's':
        if (v1 <= precio and v2 >= precio):
            return str('Ok')
            #CALCULAR PORCENTAJE
        else:
            #print(x,change,str('No'))
            return str('No')
    if rs == 'r':
        #cr0 = difOkrs(r0, 'r', bajo, alto)
        if (v1 <= precio and v2 <= precio):
            return str('Ok')
        else:
            #print(x,change,str('No'))
            return str('No')

def getDataPerDate (inicio, dataframe):
    if(dataframe is not None):
        dfcollections = {}
        dfcollection = {}
        start_date = inicio
        d = []
        dfcollection = pd.DataFrame(columns={0: 'Date', 1: 'Open', 2 : 'High', 3 : 'Low', 4 : 'Close', 5: 'Volumen'})

        for a in dataframe.itertuples(index=False, name='Pandas'):
            if( a.Date >= start_date ):
                d.append(a)

        dfcollection = pd.DataFrame(d)
        # print(dfcollection[e].info())

        return dfcollection

acp = []
acs0 = []
acs1 = []
acs2 = []
acs3 = []
acr0 = []
acr1 = []
acr2 = []
acr3 = []
acr23 = []
arrexp = []
def parsearDate(a):
    f = pd.to_datetime(a, unit='ms', format=None).to_pydatetime() - timedelta(hours=3)#%H:%M:%S
    f = f.strftime("%Y-%m-%d %H:%M:%S")
    return str(f)

def fibocomp(dataframe,s0,s1,s2,s3,r0,r1,r2,r3,p,a,f):
    if( dataframe is not None and dataframe.empty is not True):
        dataframe.Low = dataframe.Low.astype(float)
        dataframe.Close = dataframe.Close.astype(float)
        dataframe.High = dataframe.High.astype(float)

        l = dataframe.Low.min()
        if( l is not None ):
            #print('Min', l)
            bajo = l

        ld = dataframe.loc[(dataframe.Low == l)].Date.iloc[0]
        if( ld is not None ):
            nf = getDataPerDate(ld, dataframe)
            if (nf is not None and nf.empty is not True):
                f1 = nf.Date.iloc[0]
                h = nf.Close.max()
                #print(str(fechaFormat_fiboind(f1)) , 'Max', h)
                alto = h
        #print('B', parsearDate(dataframe.Date.max()))
        #print('C', dataframe.loc[(dataframe.Close == h)])
        #print('D', dataframe.loc[(dataframe.Close == h)].Date.item())
        #print(dataframe.Close[0], round(a,7))
        #print(str(dataframe.Close[0]),' == ',str(float(round(a,7))),(dataframe.Close[0] == float(round(a,7))))
        #print(dataframe)
        '''if (dataframe.Close[0] == float(round(a,7))):
            dactual = dataframe.loc[(dataframe.Close == float(round(a,7)))].Date.iloc[0]
            print(str(fechaFormat_fiboind(dactual)))
        else:
        '''
        #dactual = dataframe.Date[0]
        if(dataframe is not None):
            #nactual = getDataPerDate(dactual, dataframe)
            if(dataframe is not None and dataframe.empty is not True):
               #f2 = nactual.Date.iloc[0]
               hactual = dataframe.High.max()
               f3 = dataframe.loc[(dataframe.High == hactual)].Date.iloc[0]
               print('' ,str(fechaFormat_fiboind(f3)) , 'Precio Mas Alto:', hactual, '<', str(float(round(a,8))), (dataframe.Close[0] < float(round(a,8))))
               aalto = hactual
            else:
               aalto = 0

        #print('bajo',bajo)
        #print('alto',alto)

        dfm = math.ceil((( round(aalto,8)  / float(round(a,8)) * 100 ) - 100))
        print('Diferencia Maxima', dfm, str(round(aalto,8)), str(float(round(a,8))))

        #if (aalto is not 0 ):
            #print( round(aalto,8), round(a,8), round(dfm,2) )
        cs0 = difOkrs(s0, 's', bajo, alto)
        cs1 = difOkrs(s1, 's', bajo, alto)
        cs2 = difOkrs(s2, 's', bajo, alto)
        cs3 = difOkrs(s3, 's', bajo, alto)
        cr0 = difOkrs(r0, 'r', bajo, alto)
        cr1 = difOkrs(r1, 'r', bajo, alto)
        cr2 = difOkrs(r2, 'r', bajo, alto)
        cr3 = difOkrs(r3, 'r', bajo, alto)
        cp = difOkrs(p, 's', bajo, alto)
        r230 = (r2*Decimal(1.03))
        cr230 = difOkrs(r230, 'r', bajo, alto)

        aexp = (a*Decimal(1.04))
        caexp = ( dfm >= 5 ) #difOkrs(aexp, 'r', a, aalto)
        print('Comprobacion Maxima', dfm , 'Mayor a 5', caexp)
        #print(str(fechaFormat_fiboind(f)),' Actual:',str.format('{0:.8f}',a),' +5% ', str.format('{0:.8f}',aexp), str.format('{0:.8f}',aalto))

        if(cp == 'Ok'):
            acp.append(cp)

        if(cs0 == 'Ok'):
            acs0.append(cs0)

        if(cs1 == 'Ok'):
            acs1.append(cs1)

        if(cs2 == 'Ok'):
            acs2.append(cs2)

        if(cs3 == 'Ok'):
            acs2.append(cs3)

        if(cr0 == 'Ok'):
            acr0.append(cr0)

        if(cr1 == 'Ok'):
            acr1.append(cr1)

        if(cr2 == 'Ok'):
            acr2.append(cr2)

        if(cr3 == 'Ok'):
            acr3.append(cr3)

        if(cr230 == 'Ok'):
            acr23.append(cr230)

        if(caexp == True):
            arrexp.append(caexp)

        comp_2 = (dfm >= 2)
        fc = {
                'P_COMP':cp,
                'S0_COMP':cs0,
                'S1_COMP':cs1,
                'S2_COMP':cs2,
                'S3_COMP':cs3,
                'R0_COMP':cr0,
                'R1_COMP':cr1,
                'R2_COMP':cr2,
                'R3_COMP':cr3,
                'R2_30_COMP':cr230,
                'Min.':bajo,
                'Max.':alto,
                'R2_30': r230,
                'Cant_S3':len(acs3),
                'Cant_R2':len(acr2),
                'Cant_R2':len(acp),
                'MAX_COMP':caexp,
                'PRECIO_MAX':aalto,
                'COMP_DFM':dfm,
                'COMP_2': comp_2
                }
        #return cs1,cs2,cs3,cr1,cr2,cr3,alto,len(acs3),len(acr2),bajo,len(acp),len(cs0),len(cs1),len(cs2),len(cs3),len(cr0),len(cr1),len(cr2),len(cr3),cp,cs0,cr0,cr23,cgz,caexp,aalto,dfm
        return fc

def get_fibo(dataframe):
    if( dataframe is not None and dataframe.empty is not True):
        price_min = Decimal(1000000)
        price_max = Decimal(0)
        for day in dataframe.itertuples(index=True, name='Pandas'):
            if Decimal(day.Close) < price_min:
                price_min = Decimal(day.Close)
            if price_max < Decimal(day.High):
                price_max = Decimal(day.High)

        actual = Decimal(dataframe['Close'].iloc[-1])
        factual = dataframe['Date'].iloc[-1]
        #Pivot Point (P) = (High + Low + Close)/3
        p = (price_max + price_min + actual)/3
        #Support 1 (S1) = P - {.382 * (High - Low)}
        #Support 2 (S2) = P - {.618 * (High - Low)}
        #Support 3 (S3) = P - {1 * (High - Low)}
        #Resistance 1 (R1) = P + {.382 * (High - Low)}
        #Resistance 2 (R2) = P + {.618 * (High - Low)}
        #Resistance 3 (R3) = P + {1 * (High - Low)}
        #Fibonacci Levels considering original trend as upward move
        #diff = Decimal(price_max) - Decimal(price_min)
        #level1 = Decimal(price_max) - Decimal(0.236) * Decimal(diff)
        #level2 = Decimal(price_max) - Decimal(0.382) * Decimal(diff)
        #level3 = Decimal(price_max) - Decimal(0.5) * Decimal(diff)
        #level4 = Decimal(price_max) - Decimal(0.618) * Decimal(diff)
        s0 = p - ( Decimal(0.233) * ( Decimal(price_max) - Decimal(price_min) ) )
        s1 = p - ( Decimal(0.382) * ( Decimal(price_max) - Decimal(price_min) ) )
        s2 = p - ( Decimal(0.618) * ( Decimal(price_max) - Decimal(price_min) ) )
        s3 = p - ( Decimal(1) * ( Decimal(price_max) - Decimal(price_min) ) )
        r0 = p + ( Decimal(0.144) * ( Decimal(price_max) - Decimal(price_min) ) )
        r1 = p + ( Decimal(0.382) * ( Decimal(price_max) - Decimal(price_min) ) )
        r2 = p + ( Decimal(0.618) * ( Decimal(price_max) - Decimal(price_min) ) )
        r3 = p + ( Decimal(1) * ( Decimal(price_max) - Decimal(price_min) ) )

        #diferencia = getDifference(dataframe)
        #change1 = round((level1*100/actual)-100,2)
        #change2 = round((level2*100/actual)-100,2)
        #change3 = round((level3*100/actual)-100,2)
        #change4 = round((level4*100/actual)-100,2)
        #okch1 = difOk(diferencia, change1)
        #okch2 = difOk(diferencia, change2)
        #okch3 = difOk(diferencia, change3)
        #okch4 = difOk(diferencia, change4)
        expo = round((r1*100/s2)-100,2)
        fibo = { 'Actual' : actual, 'Fecha' : factual, 'Min': price_min, 'Dif': expo, 'P' : p, 'S0' : s0 , 'S1' : s1, 'S2': s2, 'S3' : s3, 'R0' : r0, 'R1' : r1, 'R2' : r2, 'R3' : r3 }
        return fibo

def WILLR(dataframe):
    w = ta.WILLR(dataframe.High, dataframe.Low, dataframe.Close, timeperiod=14)
    return w

def fechaFormat_fiboind(fecha):
    UTC_HAN = pd.to_datetime(fecha, unit='ms', format=None).to_pydatetime() + timedelta(hours=3)#%H:%M:%S
    UTC_HAN = UTC_HAN.strftime("%d-%m-%Y %H:%M:%S")
    return str(UTC_HAN)

a = 0
def fechaUTC_format(fecha):
    #print(a)
    #orig 2021-12-29 09:00:00 2021-12-28 17:00:00
    t = datetime.strptime(fecha, '%d %B, %Y %H:%M:%S')
    UTC1 = t - timedelta(hours=5)
    UTC2 = t + timedelta(hours=3)
    UTC1_3 = UTC1 - timedelta(hours=3)
    UTC2_3 = UTC2 - timedelta(hours=3)
    #print('orig',str(UTC1),str(UTC2))
    #print('orig_FORMAT',str(UTC1_3),str(UTC2_3))
    if (a == 1):
        #print('orig',str(UTC1_3),str(UTC2_3))
    return str(UTC1),str(UTC2)

def fechaFN(fecha):
    t = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
    UTC_FN = pd.to_datetime(t).to_pydatetime() + timedelta(days=1)
    UTC_FN = UTC_FN.strftime("%d-%m-%Y")
    return str(UTC_FN)

b = 0
def fechaUTC_fibo(fecha):
    #print(b)
    t = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
    UTC = t - timedelta(days=14, hours=3)
    UTC1_3 = UTC - timedelta(hours=3)
    UTC2_3 = t - timedelta(hours=3)
    if (b == 1):
        #print('RSI/',str(UTC1_3), str(UTC2_3))
    return str(UTC)

def fechaUTC_rev(fecha):
    t = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
    UTC = t + timedelta(hours=2)
    UTC1 = UTC + timedelta(hours=6)
    #UTC2_3 = t - timedelta(hours=3)
    #if (b == 1):
    #    print('RSI/',str(UTC1_3), str(UTC2_3))
    return str(UTC), str(UTC1)

def fecha_comp(fecha):
    #print(v)
    t = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
    UTC1 = t + timedelta(hours=8)
    UTC2 = UTC1 + timedelta(hours=7)
    UTC1_3 = UTC1 - timedelta(hours=3)
    UTC2_3 = UTC2 - timedelta(hours=3)
    if (v == 1):
        #print('comp',str(UTC1_3),str(UTC2_3))
    #print('Comp no format',str(UTC1),str(UTC2))
    #print('Comp formateada',str(UTC1_3),str(UTC2_3))
    return str(UTC1),str(UTC2)

v=0
def fechaUTC_comp(fecha):
    #print(v)
    t = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
    UTC1 = t #+ timedelta(days=0, hours=7)
    UTC2 = t + timedelta(hours=6)
    UTC1_3 = UTC1 - timedelta(hours=3)
    UTC2_3 = UTC2 - timedelta(hours=3)
    if (v == 1):
        #print('comp',str(UTC1_3),str(UTC2_3))

    #print('comp',str(UTC1),str(UTC2))
    return str(UTC1),str(UTC2)

def getRSI(dataframe):
    if( len(dataframe) > 0):
        # precios = histo['Close'].apply(Decimal)
        # print(precios) #= histo['Close'].apply(Decimal)
        indiceRSI = ta.RSI(dataframe['Close'], timeperiod=14) # <<<< PROBLEMA RSI LIBRERIA TA
        #print("functionRSI",indiceRSI)
        indiceRSI = indiceRSI.iloc[-1]
        if (indiceRSI is not None and math.isnan(indiceRSI) is not True):
            return math.floor(indiceRSI)
        else:
            return 0

def reversal(dataframe):
    if( dataframe is not None and dataframe.empty is not True and len(dataframe) >= 6):
        dataframe.Open = dataframe.Open.astype(np.float64)
        dataframe.High = dataframe.High.astype(np.float64)
        dataframe.Low = dataframe.Low.astype(np.float64)
        dataframe.Close = dataframe.Close.astype(np.float64)
        dataframe.Volumen = dataframe.Volumen.astype(np.float64)
        eng = ta.CDLENGULFING(dataframe.Open,dataframe.High,dataframe.Low,dataframe.Close)
        #print('---------------------------')
        #print(eng)
        #print('---------------------------')
        harami = ta.CDLHARAMI(dataframe.Open,dataframe.High,dataframe.Low,dataframe.Close)
        mbozu_length = ta.CDLKICKINGBYLENGTH(dataframe.Open,dataframe.High,dataframe.Low,dataframe.Close)
        mbozu_close = ta.CDLCLOSINGMARUBOZU(dataframe.Open,dataframe.High,dataframe.Low,dataframe.Close)
        mbozu = ta.CDLMARUBOZU(dataframe.Open,dataframe.High,dataframe.Low,dataframe.Close)
        a = 4
        aroon = ta.AROON(dataframe.High,dataframe.Low,4)
        #print(mbozu_length)
        #print(mbozu_close)
        mb = False
        mbi = 0
        for l, v in enumerate(mbozu_close):
            if( v != 0 ):
                mb = True
                mbi = l

        #print(eng)
        b = False
        #i = -1
        #e = -1
        for idx, val in enumerate(eng):
            if( val == -100 ):
                #print('>>>',idx,'True')
                b = True

        #print(harami)
        ##print(aroon)
        aroon_up = aroon[0].iloc[-1]
        aroon_down = aroon[1].iloc[-1]
        #print('Up', aroon[0])
        #print('Down', aroon[1])
        result = { 'Aroon_Up' : aroon_up, 'Aroon_Down': aroon_down, 'ENG_Bool' : b, 'MBOZU_Bool' : mb, 'ENG_17': eng.iloc[-5], 'ENG_18': eng.iloc[-4],'ENG_19': eng.iloc[-3],'ENG_20': eng.iloc[-2],'ENG_21': eng.iloc[-1],'MBOZU_17': mbozu_close.iloc[0], 'MBOZU_18': mbozu_close.iloc[1],'MBOZU_19': mbozu_close.iloc[2],'MBOZU_20': mbozu_close.iloc[3],'MBOZU_21': mbozu_close.iloc[4]}
        return result
        #if(harami is not None and len(harami) == 15):
        #    if (harami.iloc[11] == -100):
        #        a = 0
        #    if (harami.iloc[12] == -100):
        #        a = 1
        #    if (harami.iloc[13] == -100):
        #        a = 2
        #    if (harami.iloc[14] == -100):
        #        a = 3
        #    return a


def CDL3WHITESOLDIERS(dataframe):
    if( dataframe is not None and dataframe.empty is not True):
        dataframe.Open = dataframe.Open.astype(np.float64)
        dataframe.High = dataframe.High.astype(np.float64)
        dataframe.Low = dataframe.Low.astype(np.float64)
        dataframe.Close = dataframe.Close.astype(np.float64)
        dataframe.Volumen = dataframe.Volumen.astype(np.float64)
        res = ta.CDL3WHITESOLDIERS(dataframe.Open.values, dataframe.High.values, dataframe.Low.values, dataframe.Close.values)
        return pd.DataFrame({'CDL3WHITESOLDIERS': res}, index=dataframe.index)

def EMA(dataframe):
    if(len(dataframe) > 3):
        dataframe.Close = dataframe.Close.astype(float)
        #dataframe.Close = np.round(dataframe.Close,8)
        #print(dataframe.Close)
        EMA = ta.EMA(dataframe.Close.values, 12)
        EMA = round(EMA[-1],2)
        #print(signal[-1])
        #print(signal)
        #print(macd.iloc[-1], signal.iloc[-1])
        return EMA

def distance(x, y):
    if x >= y:
        result = x - y
    else:
        result = y - x
    return result

def CCI(dataframe):
    if( dataframe is not None and dataframe.empty is not True and len(dataframe) > 6):
        dataframe.Open = dataframe.Open.astype(np.float64)
        dataframe.High = dataframe.High.astype(np.float64)
        dataframe.Low = dataframe.Low.astype(np.float64)
        dataframe.Close = dataframe.Close.astype(np.float64)
        dataframe.Volumen = dataframe.Volumen.astype(np.float64)
        CCI = ta.CCI(dataframe.High, dataframe.Low, dataframe.Close, 8)
        OBV = ta.OBV(dataframe.Close, dataframe.Volumen)
        CCI_1 = CCI.iloc[-1]
        CCI_2 = CCI.iloc[-2]
        CCI_3 = CCI.iloc[-3]

        #print(OBV)
        CCI_D = distance(round(CCI_1,2) , round(CCI_2,2))
        #print(CCI_D)
        CCI_DIF = False
        if (CCI_D > 30):
            CCI_DIF = True
        #print('CCI', CCI_1, CCI_2, CCI_3)
        return round(CCI_1,2), round(CCI_2,2), CCI_DIF, round(CCI_3,2)

def CMO(dataframe):
    if( dataframe is not None and dataframe.empty is not True and len(dataframe) > 3):
        #print("SI CMO", len(dataframe))
        dataframe.Open = dataframe.Open.astype(np.float64)
        dataframe.High = dataframe.High.astype(np.float64)
        dataframe.Low = dataframe.Low.astype(np.float64)
        dataframe.Close = dataframe.Close.astype(np.float64)
        CMO = ta.CMO(dataframe.Close, timeperiod=12)
        #print('CMO',CMO)
        if (CMO is not None):
            return round(CMO.iloc[-1],2)

        else:
            return 0
    else:
        return 0

def iSTOCH(dataframe):
    #print('FUNC STOCHASTIC')
    #print('Lenght', len(dataframe))
    if( dataframe is not None and dataframe.empty is not True and len(dataframe) > 6):
        #print('DATAFRAME OK')
        dataframe.Open = dataframe.Open.astype(np.float64)
        dataframe.High = dataframe.High.astype(np.float64)
        dataframe.Low = dataframe.Low.astype(np.float64)
        dataframe.Close = dataframe.Close.astype(np.float64)
        slowk, slowd = ta.STOCH(dataframe.High, dataframe.Low, dataframe.Close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        #print('>>>>  slowk',slowk.iloc[-3])
        #print('>>>>  slowk',slowk.iloc[-2])
        #print('>>>>  slowk',slowk.iloc[-1])
        #print('slowd',slowd.iloc[-1])
        if (slowk is not None and slowd is not None):
            res = [ [float(slowk.iloc[-3]), float(slowd.iloc[-3])], [float(slowk.iloc[-2]), float(slowd.iloc[-2])], [float(slowk.iloc[-1]), float(slowd.iloc[-1])] ]
    else:
        res = [[0,0],[0,0],[0,0]]
    revstoch20 = []
    for val in res:
        #print('v',val, (val[0]/val[1]), (val[0]/val[1]) >= 0.96)
        if (val[0] is not 0 and val[1] is not 0 and float(val[0]/val[1]) >= 0.96 and float(val[0]/val[1]) <= 1.1):
            revstoch20.append(True)
        else:
            revstoch20.append(False)

    #print('STOCHASTICS RSI 19 //', 'Actual:', str.format('{0:.1f}',res[0][0]), 'Down:', str.format('{0:.1f}',res[0][1]), '/',  (res[0][0]/res[0][1]))
    #print('STOCHASTICS RSI 20 //', 'Actual:', str.format('{0:.1f}',res[1][0]), 'Down:', str.format('{0:.1f}',res[1][1]), '/',  (res[1][0]/res[1][1]))
    #print('STOCHASTICS RSI 21 //', 'Actual:', str.format('{0:.1f}',res[2][1]), 'Down:', str.format('{0:.1f}',res[2][1]), '/',  (res[2][0]/res[2][1]))
    #print('STOCHASTICS RSI', revstoch20)
    return res, revstoch20

def MOM(dataframe):
    if( dataframe is not None and dataframe.empty is not True and len(dataframe) > 3):
        mom = ta.MOM(dataframe, timeperiod=12)
        if (mom is not None):
            return round(mom.iloc[-1],2)
        else:
            return 0
    else:
       return 0

def ROC(dataframe):
    if( dataframe is not None and dataframe.empty is not True and len(dataframe) > 3):
        rateofchange_12 = ta.ROC(dataframe, timeperiod=12)
        rateofchange_3 = ta.ROC(dataframe, timeperiod=3)
        rateofchange_1 = ta.ROC(dataframe, timeperiod=1)
        if (rateofchange_12 is not None):
            return round(rateofchange_12.iloc[-1],2), round(rateofchange_3.iloc[-1],2), round(rateofchange_1.iloc[-1],2)
        else:
            return 0
    else:
       return 0

def getVolumen(dataframe):
    if( len(dataframe) > 5 ):
        dataframe.Volumen = dataframe.Volumen.astype(np.float64)
        #volumen = float(0)
        #for day in dataframe.itertuples(index=True, name='Pandas'):
        #    volumen = volumen + day.Volumen
        promVolumen = (dataframe.Volumen.iloc[-2] + dataframe.Volumen.iloc[-1]) / 2
        ##promVolumen = volumen / len(dataframe)
        #print(promVolumen)
        return promVolumen

    else:
        return 10

mediasrotas = []
senalesdata = []
hoy = date.today()
f_hoy1 = hoy.strftime("%d/%m/%Y")
#edf = pd.DataFrame(columns = ['Ticker','Coef','Delta','Actual','Min','Dif','CH2','Change2','Level2','COMP_CH2','CH3','Change3','Level3','COMP_CH3'])
def crearImagen(senales):
    W, H = (1080, 1920)
    hoy = date.today()
    f_hoy1 = hoy.strftime("%d/%m/%Y")
    f_file = hoy.strftime("%d_%m_%Y")
    #ticker = '$'+ticker
    #porcentaje = '~%'+'10' #dif
    #probable = price1
    #probable2 = price2

    img_url = os.path.join(dir, 'background18hs.jpg')
    img = Image.open(img_url).convert('RGB')
    draw = ImageDraw.Draw(img)

    font_url = os.path.join(dir, 'fonts', 'OpenSans_Condensed-SemiBold.ttf')
    font = ImageFont.truetype(font_url, 48)
    font0_url = os.path.join(dir, 'fonts', 'OpenSans_Condensed-Bold.ttf')
    font0 = ImageFont.truetype(font0_url, 50)
    font1_url = os.path.join(dir, 'fonts', 'OpenSans_Condensed-SemiBold.ttf')
    font1 = ImageFont.truetype(font1_url, 36)
    font2_url = os.path.join(dir, 'fonts', 'OpenSans-LightItalic.ttf')
    font2 = ImageFont.truetype(font2_url, 70)
    draw.text((640, 320), f_hoy1, (255, 255, 255), font=font2)
    m1 = 425
    m2 = 435
    ee=0
    for s in senales:
        if (ee < 15):
            m1 = m1 + 75
            m2 = m2 + 75
            ticker = '$'+str(s[0])
            diferencia = '~%'+s[1]
            draw.text((80, m1), ticker, (255, 255, 255), font=font)
            draw.text((400, m2), str(diferencia), (199, 211, 0), font=font1)
            draw.text((530, m1), str(s[2]), (255, 255, 255), font=font0)
            draw.text((785, m1), str(s[3]), (255, 255, 255), font=font0)
            ee+=1

    str_fname = 'signals_'+f_file+'.jpg'
    #str_fname_url = urljoin('',str_fname)
    str_file_path = str(dir+str_fname)
    mediasrotas.append(str_file_path)
    # saving the image
    img.save(str_file_path)

#edf = pd.DataFrame(columns = ['Ticker','Coef','Delta','3WS','MACD','MA','Rev','RSI','Vol','Actual','Max','Min','DifMax','Dif','S0R0','P','COM_P','S0','S0_COM','S1','COM_S1','S2','COM_S2','S3','COM_S3','R0','DIF_R0','COM_R0','R1','DIF_R1','COM_R1','R2','DIF_R2','COM_R2','R3','COM_R3','R23','DIF_R23','COM_R23'])
edf = pd.DataFrame(columns = ['Ticker','Coef','COMP_2','Delta','EMA','MA','CCI-1','CCI-2','AROON_UP','AROON_DOWN','ROC','MOM','CMO','STOCH_K','STOCH_D','RSI','Vol','Max','Actual','5%','Max','Min','DifR2','Dif_High','P','S0','S1','S2','S3','R0','R1','R2','R3'])

e=[]
hoy = date.today()
fecha_hoy = hoy.strftime("%d %B, %Y 18:00:00")

signals =  []
signals.append(':bangbang: Utiliza estas señales a tu bajo tu propia responsabilidad.')
signals.append(':clock2: El mejor rendimiento se obtiene de las 2am hacia las 10am.')
signals.append('-----------------------------------------')

def randMoneda(dataframe):
    print('Dataframe Lenght randMoneda', len(dataframe))
    if ( len(dataframe) > 3):
        randomness = []
        RANDOM1 = random.SystemRandom()
        E = RANDOM1.choice(dataframe)
        RANDOM2 = random.SystemRandom()
        A = RANDOM2.choice(dataframe)
        randomness.append(E[0])
        randomness.append(A[0])
        return randomness

c_2 = []
rMoneda = []
cmo_count = 0
for stock in df['symbol']:
    a += 1
    tweet = []
    intervalo = Client.KLINE_INTERVAL_1HOUR
    #COEF-DELTA>MIE-VIE
    f = fibo_fechas.f_format(fecha_hoy)
    #f = fibo_fechas.f_format("29 December, 2021 18:00:00")
    hist = get_prices(stock, intervalo, f[0], f[1])
    # print('for',type(hist))
    if (hist is not None):
            if( len(hist) > 3):
                delta = getDelta(hist)
                promVolumen = getVolumen(hist)
                fiboind = get_fibo(hist)
                dif = ( ( fiboind['R1'] / fiboind['S0'] ) * 100 ) - 100
                start_rev = fibo_fechas.f_rev(f[0])
                print('---------------------------')
                hist_rev = get_prices(stock, intervalo, start_rev[0], start_rev[1])
                print(stock)
                #print('Hora REV', start_rev, f[1])
                r = reversal(hist_rev)
                #and stoch[0] > 50 and stoch[1] > 50  dif > 15): # and c[0] > 0 and iRSI > 40 and iRSI < 65 and roc > 9 and mom > -1 ): #and iCCI[0] < 100):  # and r[0] <= 50 and r[1] <= 70):  #and iCCI[0] > 150 // if( delta < 15): if( iMACD > 0 and c[0] > 0) >> %50
                if (start_rev is not None and delta is not None and r is not None and promVolumen is not None) :

                    if( delta > 5 and delta < 25 and dif > 3 and r['ENG_Bool'] == True and promVolumen > 50000):
                        start1 = fibo_fechas.f_indicators(f[0])
                        #print(start1)
                        #print(f[1])
                        intervalo1 = Client.KLINE_INTERVAL_1DAY
                        hist1 = get_prices(stock, intervalo1, start1, f[1])
                        iEMA = EMA(hist1)
                        iRSI = getRSI(hist1)
                        iMA = ta.MA(hist1.Close, timeperiod=14, matype=0)
                        iMA = round(iMA.iloc[-1],4)
                        if( iRSI is not None and iEMA is not None and iMA is not None ): #and r is not None):
                            #COMP SAB-MAR
                            v += 1
                            #hist2 = get_prices(stock, intervalo, fecha_comp[0], fecha_comp[1])
                            #di = ( fiboind['R1'] / fiboind['Actual'] * 100 ) - 100
                            difr0 = ( fiboind['R0'] / fiboind['Actual'] * 100 ) - 100
                            mom = MOM(hist1.Close)
                            roc = ROC(hist1.Close)
                            cmo = CMO(hist1)
                            stoch = iSTOCH(hist1)
                            ##EVALUAR 2HR PREVIAS >>> RSI 50~+60 / STOCH RSI 1-30~60 2-30~40 rsi1==rsi2 REVERSAL / CCI
                            iCCI = CCI(hist) ##PROBAR CON REGISTROS DE 20 DIAS
                            coef = get_coef(hist)
                            c = coef[1][0]
                            #print('t',f[0])
                            date_comp = fecha_comp(str(f[0]))
                            ##BUSCAR ENTRE COEF 4-12 Y MEJOR DELTA A LOS DOS DIAS
                            #print(start1,f[1])
                            hist3 = get_prices(stock, intervalo, date_comp[0], date_comp[1])
                            fibocom = fibocomp(hist3, fiboind['S0'], fiboind['S1'], fiboind['S2'],fiboind['S3'],fiboind['R0'],fiboind['R1'], fiboind['R2'],fiboind['R3'],fiboind['P'],fiboind['Actual'],fiboind['Fecha'])
                            #difmax = ( (Decimal(fibocom[25]) / Decimal(fiboind['Actual'])) * 100 ) - 100
                            print( str(stock),'\t',str.format('{0:.1f}',dif), str(iMA), str(coef[1][0]), str.format('{0:.2f}', delta), str.format('{0:.8f}',fiboind['Actual']), str.format('{0:.8f}',fiboind['S1']),str.format('{0:.8f}',fiboind['R0']),str.format('{0:.8f}',fiboind['R1']),str.format('{0:.8f}',fiboind['R2']),'Up',str(r['Aroon_Up']),'Down',str(r['Aroon_Down']))
                            if (cmo > 0):
                                cmo_count += 1
                                signal = ':ticket: ' + stock + '\t' + ' Buy: ' + str.format('{0:.8f}',fiboind['S0']) +' Target1: ' + str.format('{0:.8f}',fiboind['R0']) + ' Target2: '+  str.format('{0:.8f}',fiboind['R1']) + ' Estim. %'+ str.format('{0:.0f}',dif)
                                signals.append(signal)
                                tweet.append('#' + str(stock) + ' #Criptoseñal '+ f_hoy1)
                                tweet.append('Utiliza estas señales bajo tu propia responsabilidad.')
                                tweet.append('Mejor rendimiento de 2am hacia las 12am.')
                                tweet.append('🟩 Buy: ~' + str.format('{0:.8f}',fiboind['S0']) )
                                tweet.append('🟩 Target1: ~' + str.format('{0:.8f}',fiboind['R0']))
                                tweet.append('🎉 Target2: ~'+  str.format('{0:.8f}',fiboind['R1']))
                                tweet.append('Estim. ~%'+ str.format('{0:.0f}',dif))
                                tweet.append('#criptomonedas #binancearg #binance')
                                # Create a tweet
                                print('--------------------------------')
                                str_tweet = "\n". join(tweet)
                                print(str_tweet)
                                api.update_status(str_tweet)
                                senalesdata.append( [ str(stock), str.format('{0:.0f}',dif), str.format('{0:.8f}',fiboind['R0']), str.format('{0:.8f}',fiboind['R1']) ] )
                                if (fibocom is not None):
                                    if(fibocom['COMP_2'] == True):
                                        c_2.append(fibocom['COMP_2'])

                                    e.append(
                                                {
                                                'Ticker' : str(stock),
                                                'COMP_2': str(fibocom['COMP_2']),
                                                'Coef' : str(coef[1][0][0][0]),
                                                'RSI' : str(iRSI),
                                                'Vol' : str(promVolumen),
                                                'Delta' : str.format('{0:.2f}',(delta)),
                                                'EMA' : str(iEMA),
                                                'MA' : str(iMA),
                                                'CCI-15' : str(iCCI[3]),
                                                'CCI-16' : str(iCCI[1]),
                                                'CCI-17' : str(iCCI[0]),
                                                'AROON_UP' : str(r['Aroon_Up']),
                                                'AROON_DOWN' : str(r['Aroon_Down']),
                                                'ROC12' : str(roc[0]),
                                                'ROC3' : str(roc[1]),
                                                'ROC1' : str(roc[2]),
                                                'MOM' : str(mom),
                                                'CMO' : str(cmo),
                                                'ENG_13': str(r['ENG_17']),
                                                'ENG_14': str(r['ENG_18']),
                                                'ENG_15': str(r['ENG_19']),
                                                'ENG_16': str(r['ENG_20']),
                                                'ENG_17': str(r['ENG_21']),
                                                'MBOZU_13': str(r['MBOZU_17']),
                                                'MBOZU_14': str(r['MBOZU_18']),
                                                'MBOZU_15': str(r['MBOZU_19']),
                                                'MBOZU_16': str(r['MBOZU_20']),
                                                'MBOZU_17': str(r['MBOZU_21']),
                                                'STOCHASTIC_A_19' : str(stoch[0][0][0]),
                                                'STOCHASTIC_D_19' : str(stoch[0][0][1]),
                                                'STOCHASTIC_B_19' : str(stoch[1][0]),
                                                'STOCHASTIC_A_20' : str(stoch[0][1][0]),
                                                'STOCHASTIC_D_20' : str(stoch[0][1][1]),
                                                'STOCHASTIC_B_20' : str(stoch[1][1]),
                                                'STOCHASTIC_A_21' : str(stoch[0][2][0]),
                                                'STOCHASTIC_D_21' : str(stoch[0][2][1]),
                                                'STOCHASTIC_B_21' : str(stoch[1][2]),
                                                'Actual' : str.format('{0:.8f}',fiboind['Actual']),
                                                '5%' : fibocom['R2_30_COMP'],
                                                'DifS0R1' : str.format('{0:.0f}',dif),
                                                'High%' : str.format('{0:.0f}',fibocom['COMP_DFM']),
                                                'R1_COMP' : str(fibocom['R1_COMP']),
                                                'R0_COMP' : str(fibocom['R0_COMP']),
                                                'P' : str.format('{0:.8f}',fiboind['P']),
                                                'S0' : str.format('{0:.8f}',fiboind['S0']),
                                                'S1' : str.format('{0:.8f}',fiboind['S1']),
                                                'S2' : str.format('{0:.8f}',fiboind['S2']),
                                                'S3' : str.format('{0:.8f}',fiboind['S3']),
                                                'R0' : str.format('{0:.8f}',fiboind['R0']),
                                                'R1' : str.format('{0:.8f}',fiboind['R1']),
                                                'R2' : str.format('{0:.8f}',fiboind['R2']),
                                                'R3' : str.format('{0:.8f}',fiboind['R3']),
                                                }
                                            )

#randomcrypto = randMoneda(senalesdata)
#res_c_2 = pd.DataFrame(c_2)
edf = pd.DataFrame(e)
print(edf)
signals.append('-----------------------------------------')
signals.append(':ok_hand: Compartir es apreciar. Propinas... :weary::wink: ')
signals.append(':white_check_mark: LINK_COINBASE')
#crearImagen(senalesdata)

#'5%' : fibocom[24],
#'Dif_High' : str.format('{0:.0f}',difmax),

#res_comp_2 = edf.loc[edf.COMP_2, True]
#print('Cantidad de Valores positivos en COMP_2 ',len(res_c_2))
#'R1_COMP' : str(fibocom[3]),
#print('Cantidad de CMO', cmo_count, len(res_c_2))

#if (len(res_c_2) > 1):
#    porcentaje_comp_2 = round(((len(res_c_2)*100/cmo_count)),2)

#dR0 = round((len(acr0)/len(edf)*100),2)
#amas5 = round((len(arrexp)/len(edf)*100),2)
#print('A+5%',amas5)
#print('R0',dR0)
#print('R1',dR1)
#print('Probabilidad de rendimiento del 2'+'%'+' es del '+' %'+str(porcentaje_comp_2))
csv_url = os.path.join(dir,'csv', '__FIBO18HS_'+fechaFN(f[0])+'.csv')
edf.to_csv(csv_url, index=False)
fs_filepath = csv_url
fs_filepath_name = ('__FIBO18HS_' + fechaFN(f[0]) + '.csv')

store_params = {
    "mimetype": "text/csv"
}
#new_filelink = fs_client.upload(filepath=fs_filepath, store_params=store_params)
#print(new_filelink.url)  # 'https://cdn.filestackcontent.com/FILE_HANDLE'
str_signals = "\n".join(signals)
url = "https://hooks.slack.com/services/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"
webhook = WebhookClient(url)
response = webhook.send(
    text="fallback",
    blocks=[
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": str_signals

            }
        }
    ]
)
#print(type(randomcrypto))
#if(randomcrypto is not None):
#    str_randomsignals = "\n".join(randomcrypto)
#    print(type(str_randomsignals))
#    webhook1 = WebhookClient(url)
#    response1 = webhook.send(
#        text="fallback",
#        blocks=[
#            {
#               "type": "section",
#                "text": {
#                    "type": "mrkdwn",
#                    "text": str_randomsignals
#                }
#            }
#        ]
#    )

# ID of channel that you want to upload file to
#channel_id = "C02MLBQSCLF"

#cl = Cl()
#cl.login('feamoneda', 'SWMW@@6@2021!!')
#def crearPhoto(media_path):
#    for file_path in media_path:
#        cl.photo_upload_to_story(file_path)

#crearPhoto(mediasrotas)

my_file = { 'file' : (csv_url, 'csv') }
url = "https://slack.com/api/files.upload"

headers = CaseInsensitiveDict()
headers = {
    "Authorization" : "Bearer xoxb-2772745627024-2844989989255-0gEfon6PFK15kS1YIQk4HldE",
    "file" : fs_filepath,
    "channels" : "C02MLBQSCLF"
}

#resp = requests.get(url, headers=headers, files=my_file)

# The name of the file you're going to upload
file_name = fs_filepath_name
# ID of channel that you want to upload file to
channel_id = "C02MLBQSCLF"

try:
    # Call the files.upload method using the WebClient
    # Uploading files requires the `files:write` scope
    result = clientSlack.files_upload(
        channels=channel_id,
        initial_comment="Here's my file :smile:",
        file=file_name,
    )
    # Log the result
    logger.info(result)

except SlackApiError as e:
    logger.error("Error uploading file: {}".format(e))

#print(resp.status_code)

# ID of channel that you want to upload file to
channel_id = "C02MLBQSCLF"