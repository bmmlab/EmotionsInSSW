# Import statements
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import statsmodels.tsa.stattools as ts
from datetime import datetime as dt  # will need this for sessions 21...24 (PB)

# Declare a function that will change owner IDs from markets (Flex-E-Markets) output from 8038,... to 1,2,...
def g1(x):
    x = x - 4563
    return x


def g2(x):
    x = x - 8037
    return x

################################################################################
# PART 1; PROCESS MARKETS OUTPUT DATA, GENERATE BID/ASK SPREADS, HOLDINGS CHANGES
# ETC.
################################################################################


# This method is called on line 576ff FOR ALL SESSIONS
def Market_data_analysis(s):  # session s (range:1 to 11 & 21 to 24) 

    d = {}
    d[s] = pd.read_csv('Official ' + str(s) + '.csv')
    df = pd.DataFrame(d[s])
    if 20 >= s >= 6:
        df['owner'] = df['owner'].apply(g1)
    elif s >= 21:
        df['owner'] = df['owner'].apply(g2)
    # PB Trading data from Flex-e-Markets for s = 6,...,11 have original owner ID numbers
    df2 = df[df['trade'] == 1]
    df2 = df2[['owner', 'order', 'consumer', 'trade', 'buy/sell']]
    df3 = pd.merge(df, df2, left_on='consumer', right_on='order', how='left')
    df4 = df3[df3['trade_y'] == 1]
    df4 = df4[['owner_x', 'order_x', 'trade_x', 'buy/sell_x']]
    mergeDf_v2 = pd.merge(df3, df4, left_on='consumer_x', right_on='order_x', how='left')
    # Uncommnent this to check merges
    mergeDf_v2.to_csv('saveit_official ' + str(s) + ' v2.csv')
    del df2
    del df3
    del df4

    def trader_trade_dummy_v2(s):
        # print(s)
        trade = []
        for l in range(len(mergeDf_v2)):
            # print(l) # PB Included to find a mistake in the order data
            a = mergeDf_v2['trade_x_y'].loc[l] == 0
            b = mergeDf_v2['owner_x_y'][l] == s
            if a and b:
                trade.append(1)
            else:
                trade.append(0)

        trade_1 = []
        for l in range(len(mergeDf_v2)):
            a = int(mergeDf_v2['trade_x_x'].loc[l]) == 1
            b = mergeDf_v2['owner_x_x'][l] == s
            if a and b:
                trade_1.append(1)
            else:
                trade_1.append(0)

        temp = [x + y for x, y in zip(trade, trade_1)]
        return temp

    # trader_trade_dummy_v2(1) *that is right*
    mergeDf_v2['trade_dummy 1'] = trader_trade_dummy_v2(1)
    mergeDf_v2['trade_dummy 2'] = trader_trade_dummy_v2(2)
    mergeDf_v2['trade_dummy 3'] = trader_trade_dummy_v2(3)
    mergeDf_v2['trade_dummy 4'] = trader_trade_dummy_v2(4)
    mergeDf_v2['trade_dummy 5'] = trader_trade_dummy_v2(5)
    mergeDf_v2['trade_dummy 6'] = trader_trade_dummy_v2(6)
    mergeDf_v2['trade_dummy 7'] = trader_trade_dummy_v2(7)
    mergeDf_v2['trade_dummy 8'] = trader_trade_dummy_v2(8)

    def change_holding_v2(s):
        trade_1 = []
        for l in range(len(mergeDf_v2)):
            a = int(mergeDf_v2['trade_dummy ' + str(s)][l]) == 1
            # if the seller is also the buyer of same order, equals zero
            b = mergeDf_v2['owner_x_x'][l] == int(s)
            c = mergeDf_v2['owner_x_y'][l] == int(s)
            if a and b:
                trade_1.append(mergeDf_v2['buy/sell_x_x'][l])
            if a and c:
                trade_1.append(mergeDf_v2['buy/sell_x_y'][l])
            if int(mergeDf_v2['trade_dummy ' + str(s)][l]) != 1:
                trade_1.append(0)

        temp = [x * y * z for x, y, z in zip(mergeDf_v2['trade_dummy ' + str(s)], mergeDf_v2['units'], trade_1)]
        return temp

    mergeDf_v2['change_holding_v2_1'] = change_holding_v2(1)
    mergeDf_v2['change_holding_v2_2'] = change_holding_v2(2)
    mergeDf_v2['change_holding_v2_3'] = change_holding_v2(3)
    mergeDf_v2['change_holding_v2_4'] = change_holding_v2(4)
    mergeDf_v2['change_holding_v2_5'] = change_holding_v2(5)
    mergeDf_v2['change_holding_v2_6'] = change_holding_v2(6)
    mergeDf_v2['change_holding_v2_7'] = change_holding_v2(7)
    mergeDf_v2['change_holding_v2_8'] = change_holding_v2(8)

    def individual_mispricing_v2(s):
        temp = []
        for l in range(len(mergeDf_v2)):
            if int(mergeDf_v2['trade_dummy ' + str(s)][l]) != 0:
                value = mergeDf_v2['trade_dummy ' + str(s)][l] * mergeDf_v2['mispricing'][l]
                temp.append(value)
            else:
                temp.append(np.nan)
        return temp

    mergeDf_v2['mispricing_1_v2'] = individual_mispricing_v2(1)
    mergeDf_v2['mispricing_2_v2'] = individual_mispricing_v2(2)
    mergeDf_v2['mispricing_3_v2'] = individual_mispricing_v2(3)
    mergeDf_v2['mispricing_4_v2'] = individual_mispricing_v2(4)
    mergeDf_v2['mispricing_5_v2'] = individual_mispricing_v2(5)
    mergeDf_v2['mispricing_6_v2'] = individual_mispricing_v2(6)
    mergeDf_v2['mispricing_7_v2'] = individual_mispricing_v2(7)
    mergeDf_v2['mispricing_8_v2'] = individual_mispricing_v2(8)

    mergeDf_v2.to_csv('saveit_official ' + str(s) + ' v2.csv')

    def position_odd_v2(s):
        temp = []
        for l in range(len(mergeDf_v2)):
            if l == 0:
                value = 20 + mergeDf_v2['change_holding_v2_' + str(s)][l]
                temp.append(value)
            if l > 0:
                value = temp[l - 1] + mergeDf_v2['change_holding_v2_' + str(s)][l]
                temp.append(value)
        return temp

    def position_even_v2(s):
        temp = []
        for l in range(len(mergeDf_v2)):
            if l == 0:
                value = 12 + mergeDf_v2['change_holding_v2_' + str(s)][l]
                temp.append(value)
            if l > 0:
                value = temp[l - 1] + mergeDf_v2['change_holding_v2_' + str(s)][l]
                temp.append(value)
        return temp

    mergeDf_v2['s1 holding_v2'] = pd.Series(position_odd_v2(1))
    mergeDf_v2['s2 holding_v2'] = pd.Series(position_even_v2(2))
    mergeDf_v2['s3 holding_v2'] = pd.Series(position_odd_v2(3))
    mergeDf_v2['s4 holding_v2'] = pd.Series(position_even_v2(4))
    mergeDf_v2['s5 holding_v2'] = pd.Series(position_odd_v2(5))
    mergeDf_v2['s6 holding_v2'] = pd.Series(position_even_v2(6))
    mergeDf_v2['s7 holding_v2'] = pd.Series(position_odd_v2(7))
    mergeDf_v2['s8 holding_v2'] = pd.Series(position_even_v2(8))

    del mergeDf_v2['change_holding_v2_1']
    del mergeDf_v2['change_holding_v2_2']
    del mergeDf_v2['change_holding_v2_3']
    del mergeDf_v2['change_holding_v2_4']
    del mergeDf_v2['change_holding_v2_5']
    del mergeDf_v2['change_holding_v2_6']
    del mergeDf_v2['change_holding_v2_7']
    del mergeDf_v2['change_holding_v2_8']

    def change_cash_v2(s):
        trade_1 = []
        for l in range(len(mergeDf_v2)):
            a = int(mergeDf_v2['trade_dummy ' + str(s)][l]) == 1
            # if the seller is also the buyer of same order, equals zero
            b = mergeDf_v2['owner_x_x'][l] == int(s)
            c = mergeDf_v2['owner_x_y'][l] == int(s)
            if a and b:
                trade_1.append(mergeDf_v2['buy/sell_x_x'][l])
            if a and c:
                trade_1.append(mergeDf_v2['buy/sell_x_y'][l])
            if int(mergeDf_v2['trade_dummy ' + str(s)][l]) != 1:
                trade_1.append(0)
        temp = [w * x * y * (-z) for w, x, y, z in
                zip(mergeDf_v2['trade_dummy ' + str(s)], mergeDf_v2['units'], mergeDf_v2['price'], trade_1)]
        return temp

    mergeDf_v2['s1 change cash'] = pd.Series(change_cash_v2(1))
    mergeDf_v2['s2 change cash'] = pd.Series(change_cash_v2(2))
    mergeDf_v2['s3 change cash'] = pd.Series(change_cash_v2(3))
    mergeDf_v2['s4 change cash'] = pd.Series(change_cash_v2(4))
    mergeDf_v2['s5 change cash'] = pd.Series(change_cash_v2(5))
    mergeDf_v2['s6 change cash'] = pd.Series(change_cash_v2(6))
    mergeDf_v2['s7 change cash'] = pd.Series(change_cash_v2(7))
    mergeDf_v2['s8 change cash'] = pd.Series(change_cash_v2(8))

    del mergeDf_v2['trade_dummy 1']
    del mergeDf_v2['trade_dummy 2']
    del mergeDf_v2['trade_dummy 3']
    del mergeDf_v2['trade_dummy 4']
    del mergeDf_v2['trade_dummy 5']
    del mergeDf_v2['trade_dummy 6']
    del mergeDf_v2['trade_dummy 7']
    del mergeDf_v2['trade_dummy 8']

    mergeDf_v2.to_csv('saveit_official ' + str(s) + ' v2.csv')

    div = pd.read_csv('Div_' + str(s) + '_official.csv')
    mergeDf_v2 = pd.read_csv('saveit_official ' + str(s) + ' v2.csv')
    div = pd.DataFrame(div)
    mergeDf_div = pd.merge(mergeDf_v2, div, left_on='period', right_on='Period', how='left')

    def cash_odd_v2(s):
        temp = []
        for l in range(len(mergeDf_v2)):
            if l == 0:
                value = 10000 + mergeDf_v2[s + ' change cash'][l]
                temp.append(value)
            if (l > 0 and int(mergeDf_v2['period'][l]) != int(mergeDf_v2['period'][l - 1])):
                value = temp[l - 1] + mergeDf_v2[s + ' change cash'][l] + mergeDf_v2[s + ' holding_v2'][l - 1] * \
                        mergeDf_div['Div_begin'][l]
                temp.append(value)
            if (l > 0 and int(mergeDf_v2['period'][l]) == int(mergeDf_v2['period'][l - 1])):
                value = temp[l - 1] + mergeDf_v2[s + ' change cash'][l]
                temp.append(value)
        return temp

    mergeDf_v2['s1 cash_v2'] = pd.Series(cash_odd_v2('s1'))

    def cash_even_v2(s):
        temp = []
        for l in range(len(mergeDf_v2)):
            if l == 0:
                value = 16000 + mergeDf_v2[s + ' change cash'][l]
                temp.append(value)
            if (l > 0 and int(mergeDf_v2['period'][l]) != int(mergeDf_v2['period'][l - 1])):
                value = temp[l - 1] + mergeDf_v2[s + ' change cash'][l] + mergeDf_v2[s + ' holding_v2'][l - 1] * \
                        mergeDf_div['Div_begin'][l]
                temp.append(value)
            if (l > 0 and int(mergeDf_v2['period'][l]) == int(mergeDf_v2['period'][l - 1])):
                value = temp[l - 1] + mergeDf_v2[s + ' change cash'][l]
                temp.append(value)
        return temp

    mergeDf_v2['s2 cash_v2'] = pd.Series(cash_even_v2('s2'))

    mergeDf_v2['s1 cash_v2'] = pd.Series(cash_odd_v2('s1'))
    mergeDf_v2['s2 cash_v2'] = pd.Series(cash_even_v2('s2'))
    mergeDf_v2['s3 cash_v2'] = pd.Series(cash_odd_v2('s3'))
    mergeDf_v2['s4 cash_v2'] = pd.Series(cash_even_v2('s4'))
    mergeDf_v2['s5 cash_v2'] = pd.Series(cash_even_v2('s5'))
    mergeDf_v2['s6 cash_v2'] = pd.Series(cash_even_v2('s6'))
    mergeDf_v2['s7 cash_v2'] = pd.Series(cash_odd_v2('s7'))
    mergeDf_v2['s8 cash_v2'] = pd.Series(cash_even_v2('s8'))

    del mergeDf_v2['s1 change cash']
    del mergeDf_v2['s2 change cash']
    del mergeDf_v2['s3 change cash']
    del mergeDf_v2['s4 change cash']
    del mergeDf_v2['s5 change cash']
    del mergeDf_v2['s6 change cash']
    del mergeDf_v2['s7 change cash']
    del mergeDf_v2['s8 change cash']

    mergeDf_v2.to_csv('saveit_official ' + str(s) + ' v2.csv')
    d = pd.read_csv('saveit_official ' + str(s) + ' v2.csv')
    mergeDf_v2 = pd.DataFrame(d)

    mergeDf_v2['Best Price'] = mergeDf_v2['trade price']
    # mergeDf['Best Price']=mergeDf['Best Price'].replace('xx',np.NaN)
    mergeDf_v2['Best Price'].fillna(method='bfill', inplace=True)
    mergeDf_v2['Best Price'] = mergeDf_v2['Best Price'].replace(np.NaN, int(50))  ##last period div or expected?
    mergeDf_v2['Best Price'] = mergeDf_v2['Best Price'].apply(pd.to_numeric)

    def Asset_value_market_v2(s):
        temp_1 = [x + y * z for x, y, z in
                  zip(mergeDf_v2[s + ' cash_v2'], mergeDf_v2[s + ' holding_v2'], mergeDf_v2['Best Price'])]
        return temp_1

    mergeDf_v2['Asset Value_market_v2_S1'] = pd.Series(Asset_value_market_v2('s1'))
    mergeDf_v2['Asset Value_market_v2_S2'] = pd.Series(Asset_value_market_v2('s2'))
    mergeDf_v2['Asset Value_market_v2_S3'] = pd.Series(Asset_value_market_v2('s3'))
    mergeDf_v2['Asset Value_market_v2_S4'] = pd.Series(Asset_value_market_v2('s4'))
    mergeDf_v2['Asset Value_market_v2_S5'] = pd.Series(Asset_value_market_v2('s5'))
    mergeDf_v2['Asset Value_market_v2_S6'] = pd.Series(Asset_value_market_v2('s6'))
    mergeDf_v2['Asset Value_market_v2_S7'] = pd.Series(Asset_value_market_v2('s7'))
    mergeDf_v2['Asset Value_market_v2_S8'] = pd.Series(Asset_value_market_v2('s8'))

    # =============================================================================
    # Choose Experiment Session (Read orriginal data anew)
    # =============================================================================

    df = pd.read_csv('Official ' + str(s) + '.csv')

    # =============================================================================
    # Set Variables
    # =============================================================================

    df['FinalPrice'] = df['trade price']
    df['tradetime'] = df['time secs']
    df.tradetime = df['time secs']

    # =============================================================================
    # Create Bid-Ask Spread
    # =============================================================================
    # Create new dataframe to filter bid-ask spread
    dt = df[['tradetime', 'period', 'side', 'price', 'market', 'trade', 'order', 'original', 'consumer', 'type']]
    dt = dt.sort_values(['tradetime'])
    dt = dt.reset_index(drop=True)

    # Convert nan in consumer to -1 (consumer is one of the output columns from Flex-E-Markets, indicating who consumed an order)
    dt.consumer = dt.consumer.fillna(-1)
    dt.consumer = dt.consumer.astype(int)
    # dt.consumer = dt.consumer.replace(-1, np.nan)

    # Remove orginal order with multiple units that are later splitted
    mySecurity = {}
    ms = {}

    mySecurity[1] = []
    for i in range(0, len(dt.tradetime)):
        if dt.consumer[i] == 0:
            ms[1] = np.nan
        elif dt.consumer[i] != 0:
            ms[1] = dt.price[i]
        else:
            ms[1] = np.nan
        mySecurity[1].append(ms[1])
    dt['Security1'] = mySecurity[1]

    dt = dt[dt.consumer != 0]
    dt = dt.reset_index(drop=True)

    # dt['temptime'] = dt['tradetime'].shift(+1)
    # dt.temptime = dt.temptime.replace(np.nan, datetime.time(0, 0))

    # =============================================================================
    # Create new dataframe for each security
    # =============================================================================
    security = dt

    security = security.drop(columns=['market', 'price', 'original'])
    security = security.sort_values(by=['tradetime', 'order'])
    security = security.reset_index(drop=True)

    # =============================================================================
    #
    # Bid-Ask Spread
    #
    # =============================================================================
    # step 1: split market session data into 15 period data
    d = {}
    for i in range(0, len(security.tradetime)):
        for p in range(1, 16):
            if int(security.period[i]) == p:
                d[p] = security.loc[security['period'] == p]
                d[p].index = np.arange(0, len(d[p]))

    # =============================================================================

    # Define a function for ask (p = period number)

    # =============================================================================
    # Create individual column for each bid order

    for p in range(1, 16):
        d[p] = pd.DataFrame(d[p])
    buy = pd.DataFrame()

    def B(p):
        buy = pd.DataFrame()
        for j in range(len(d[p].tradetime)):
            mySecurity = []
            for i in range(len(d[p].tradetime)):
                if d[p].side[i] == 'BUY' and i == j and d[p].order[i] < d[p].consumer[i]:
                    ms = d[p]['Security1'][i]
                elif d[p].side[i] == 'BUY' and i == j and d[p].consumer[i] == -1:
                    ms = d[p]['Security1'][i]
                elif d[p].trade[i] == 1:
                    ms = np.nan
                elif d[p].tradetime[j] == d[p].tradetime[i] and d[p].order[j] == d[p].consumer[i] and d[p].type[
                    i] == 'CANCEL':
                    ms = np.nan
                else:
                    ms = np.nan
                mySecurity.append(ms)
            buy[str(j) + 'BUY'] = mySecurity

        # Forward fill all lasting bids in market (where consumer == -1)

        for i in range(len(d[p].tradetime)):
            if d[p].side[i] == 'BUY' and d[p].consumer[i] == -1:
                buy[str(i) + 'BUY'] = buy[str(i) + 'BUY'].fillna(method='pad')

        # Find last entry index location for each bid that is cancelled
        indexcb = {}

        temp1 = []
        for i in range(len(d[p].side)):
            if d[p].side[i] == 'BUY':
                temp1 = d[p].loc[d[p]['type'] == 'CANCEL'].index.values.tolist()
        xlst = []
        for i in range(len(temp1)):
            x = d[p].consumer[temp1[i]]
            xlst.append(x)
        indexcb = pd.DataFrame()
        indexcb = {'consumer': xlst, 'indexno': temp1}
        indexcb = pd.DataFrame(indexcb)

        # Find last entry index location for each bid that is traded
        indextb = {}

        temp2 = []
        for i in range(len(d[p].side)):
            if d[p].side[i] == 'BUY':
                temp2 = d[p].loc[d[p]['trade'] == 1].index.values.tolist()
        ylst = []
        for i in range(len(temp2)):
            y = d[p].consumer[temp2[i]]
            ylst.append(y)
        indextb = pd.DataFrame()
        indextb = {'consumer': ylst, 'indexno': temp2}
        indextb = pd.DataFrame(indextb)

        # Forward fill bids between submitted and cancelled
        bid = {}

        bid = pd.DataFrame()
        maxbid = []
        for i in range(len(indexcb.consumer)):
            temp1 = []
            for j in range(len(d[p].tradetime)):
                if indexcb.consumer[i] == d[p].order[j] and d[p].side[j] == 'BUY':
                    temp1 = indexcb.indexno[i] + 1
                    buy[str(j) + 'BUY'] = buy[str(j) + 'BUY'].fillna(method='pad')[:temp1]
        for i in range(len(indextb.consumer)):
            temp2 = []
            for j in range(len(d[p].tradetime)):
                if indextb.consumer[i] == d[p].order[j] and d[p].side[j] == 'BUY':
                    temp2 = indextb.indexno[i] + 1
                    buy[str(j) + 'BUY'] = buy[str(j) + 'BUY'].fillna(method='pad')[:temp2]
        maxbid = np.array(buy.max(axis=1))

        bid['tradetime'] = d[p]['tradetime']
        bid['period'] = d[p]['period']
        bid['bid'] = maxbid

        return bid

    df_b = {}
    for p in range(1, 16):
        df_b[p] = B(p)

    bid_price = pd.concat(
        [df_b[1], df_b[2], df_b[3], df_b[4], df_b[5], df_b[6], df_b[7], df_b[8], df_b[9], df_b[10], df_b[11], df_b[12],
         df_b[13], df_b[14], df_b[15]])
    bid_price.index = np.arange(0, len(bid_price))
    # =============================================================================

    # Ask

    # =============================================================================

    sell = {}
    for p in range(1, 16):
        d[p] = pd.DataFrame(d[p])
    buy = pd.DataFrame()

    # =============================================================================
    # Define a function for ask (p = period number)
    # =============================================================================
    # Create indidivdual column for each ask order

    def A(p):

        sell = pd.DataFrame()
        for j in range(0, len(d[p].tradetime)):
            mySecurity = []
            for i in range(0, len(d[p].tradetime)):
                if d[p].side[i] == 'SELL' and i == j and d[p].order[i] < d[p].consumer[i]:
                    ms = d[p]['Security1'][i]
                elif d[p].side[i] == 'SELL' and i == j and d[p].consumer[i] == -1:
                    ms = d[p]['Security1'][i]
                elif d[p].trade[i] == 1:
                    ms = np.nan
                elif d[p].tradetime[j] == d[p].tradetime[i] and d[p].order[j] == d[p].consumer[i] and d[p].type[
                    i] == 'CANCEL':
                    ms = np.nan
                else:
                    ms = np.nan
                mySecurity.append(ms)
            sell[str(j) + 'SELL'] = mySecurity

        # Forward fill all lasting bids in market (where consumer == -1)

        for i in range(0, len(d[p].tradetime)):
            if d[p].side[i] == 'SELL' and d[p].consumer[i] == -1:
                sell[str(i) + 'SELL'] = sell[str(i) + 'SELL'].fillna(method='pad')

        # Find last entry index location for each bid that is cancelled
        indexcs = {}

        temp1 = []
        for i in range(0, len(d[p].side)):
            if d[p].side[i] == 'SELL':
                temp1 = d[p].loc[d[p]['type'] == 'CANCEL'].index.values.tolist()
        xlst = []
        for i in range(len(temp1)):
            x = d[p].consumer[temp1[i]]
            xlst.append(x)
        indexcs = pd.DataFrame()
        indexcs = {'consumer': xlst, 'indexno': temp1}
        indexcs = pd.DataFrame(indexcs)

        # Find last entry index location for each bid that is traded
        indexts = {}

        temp2 = []
        for i in range(0, len(d[p].side)):
            if d[p].side[i] == 'SELL':
                temp2 = d[p].loc[d[p]['trade'] == 1].index.values.tolist()
        ylst = []
        for i in range(len(temp2)):
            y = d[p].consumer[temp2[i]]
            ylst.append(y)
        indexts = pd.DataFrame()
        indexts = {'consumer': ylst, 'indexno': temp2}
        indexts = pd.DataFrame(indexts)

        # Forward fill bids between submitted and cancelled
        ask = {}

        ask = pd.DataFrame()
        minask = []
        for i in range(len(indexcs.consumer)):
            temp1 = []
            for j in range(0, len(d[p].tradetime)):
                if indexcs.consumer[i] == d[p].order[j] and d[p].side[j] == 'SELL':
                    temp1 = indexcs.indexno[i] + 1
                    sell[str(j) + 'SELL'] = sell[str(j) + 'SELL'].fillna(method='pad')[:temp1]
        for i in range(len(indexts.consumer)):
            temp2 = []
            for j in range(0, len(d[p].tradetime)):
                if indexts.consumer[i] == d[p].order[j] and d[p].side[j] == 'SELL':
                    temp2 = indexts.indexno[i] + 1
                    sell[str(j) + 'SELL'] = sell[str(j) + 'SELL'].fillna(method='pad')[:temp2]
        minask = np.array(sell.min(axis=1))

        ask['tradetime'] = d[p]['tradetime']
        ask['period'] = d[p]['period']
        ask['ask'] = minask

        return ask

    df_a = {}
    for p in range(1, 16):
        df_a[p] = A(p)

    ask_price = pd.concat(
        [df_a[1], df_a[2], df_a[3], df_a[4], df_a[5], df_a[6], df_a[7], df_a[8], df_a[9], df_a[10], df_a[11], df_a[12],
         df_a[13], df_a[14], df_a[15]])
    ask_price.index = np.arange(0, len(ask_price))

    bidask = {}

    bidask = pd.merge(ask_price, bid_price, left_index=True, right_index=True)

    bidask = bidask[['tradetime_x', 'ask', 'bid']]
    bidask['spread'] = bidask['ask'] - bidask['bid']

    mergeDf_v2 = pd.merge(mergeDf_v2, bidask, left_on='time secs', right_on='tradetime_x', how='left')

    mergeDf_v2.to_csv('saveit_official ' + str(s) + ' v2.csv')


# =============================================================================
# Run the Below Function for market data
# =============================================================================

Market_data_analysis(1)
Market_data_analysis(2)
Market_data_analysis(3)
Market_data_analysis(4)
Market_data_analysis(5)
Market_data_analysis(6)
Market_data_analysis(7)
Market_data_analysis(8)
Market_data_analysis(9)
Market_data_analysis(10)
Market_data_analysis(11)
Market_data_analysis(12)

Market_data_analysis(21)
Market_data_analysis(22)
Market_data_analysis(23)
Market_data_analysis(24)


################################################################################
# PART 2; PROCESS VOLUME AND EARNINGS DATA, INPUTS INLiNE
################################################################################

# =============================================================================
# Inequality Measures Dataframe (Final Holding)
# PB Before this, need to collect final holdings AND volume.
# First collect from earnings.xlsx file, then add volume
# =============================================================================
data_1 = {'ID': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8'],
        'volume': [58, 83, 42, 116, 50, 29, 37, 45],
        'earning': [20945, 16070, 16065, 29015, 23500, 22085, 22575, 17745],
         'holding':[0,7,4,60,22,17,15,3],
         'session':[1,1,1,1,1,1,1,1]}
df_1 = pd.DataFrame(data_1)
#df_1

data_2 = {'ID': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8'],
        'volume': [92, 16, 66, 98, 63, 101, 88, 76],
        'earning': [22565, 24895, 22180, 26685, 24245, 20745, 25760, 23325],
         'holding':[4,0,14,40,39,15,6,10],
         'session':[2,2,2,2,2,2,2,2]}
df_2 = pd.DataFrame(data_2)
#df_2

data_3 = {'ID': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8'],
        'volume': [97, 41, 52, 240, 383, 90, 37, 66],
        'earning': [25025, 27350, 30690, 42710, 890, 31540, 31850, 29145],
         'holding':[11,13,0,0,103,0,1,0],
         'session':[3,3,3,3,3,3,3,3]}
df_3 = pd.DataFrame(data_3)
#df_3

data_4 = {'ID': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8'],
        'volume': [38, 24, 36, 35, 42, 143, 56, 152],
        'earning': [23010, 22790, 27905, 24370, 22235, 17760, 21450, 14880],
         'holding':[6,10,2,1,20,11,32,46],
         'session':[4,4,4,4,4,4,4,4]}
df_4 = pd.DataFrame(data_4)

data_5 = {'ID': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8'],
        'volume': [23, 84, 33, 23, 70, 89, 24, 132],
        'earning': [25000, 21340, 25645, 24175, 20525, 29585, 25220, 31710],
         'holding':[13, 0, 7, 19, 4, 23, 28, 34],
         'session':[5,5,5,5,5,5,5,5]}
df_5 = pd.DataFrame(data_5)

data_6 = {'ID': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8'],
        'volume': [23, 84, 33, 23, 70, 89, 24, 132],
        'earning': [22940, 26015, 24990, 10110, 21220, 36165, 10945, 41355],
         'holding':[26, 10, 5, 21, 15, 0, 0, 50],
         'session':6*np.ones(8,dtype=int)}
df_6 = pd.DataFrame(data_6)

data_7 = {'ID': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8'],
        'volume': [23, 84, 33, 23, 70, 89, 24, 132],
        'earning': [24370, 22530, 25065, 25615, 21010, 27290, 23870, 23850],
         'holding':[53, 0, 11, 0, 40, 0, 8, 16],
         'session':7*np.ones(8,dtype=int)}
df_7 = pd.DataFrame(data_7)

data_8 = {'ID': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8'],
        'volume': [23, 84, 33, 23, 70, 89, 24, 132],
        'earning': [15910, 22125, 15540, 18575, 19595, 20470, 19580, 17005],
         'holding':[7, 7, 46, 23, 15, 1, 0, 29],
         'session':8*np.ones(8,dtype=int)}
df_8 = pd.DataFrame(data_8)

data_9 = {'ID': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8'],
        'volume': [23, 84, 33, 23, 70, 89, 24, 132],
        'earning': [15640, 30580, 24800, 25535, 15945, 30905, 35350, 24445],
         'holding':[35, 0, 5, 2, 48, 11, 0, 27],
         'session':9*np.ones(8,dtype=int)}
df_9 = pd.DataFrame(data_9)

data_10 = {'ID': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8'],
        'volume': [23, 84, 33, 23, 70, 89, 24, 132],
        'earning': [26110, 24805, 27755, 25045, 30575, 25620, 27280, 25610],
         'holding':[5, 4, 35, 1, 37, 2, 37, 7],
         'session':10*np.ones(8,dtype=int)}
df_10 = pd.DataFrame(data_10)

data_11 = {'ID': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8'],
        'volume': [23, 84, 33, 23, 70, 89, 24, 132],
        'earning': [24145, 27550, 25700, 22165, 21770, 24640, 21670, 29160],
         'holding':[6, 1, 24, 18, 18, 43, 14, 4],
         'session':11*np.ones(8,dtype=int)}
df_11 = pd.DataFrame(data_11)

data_12 = {'ID': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8'],
        'volume': [23, 84, 33, 23, 70, 89, 24, 132],
        'earning': [24500, 26925, 27260, 21720, 28710, 32920, 30200, 26965],
         'holding':[4, 16, 19, 2, 9, 22, 37, 19],
         'session':12*np.ones(8,dtype=int)}
df_12 = pd.DataFrame(data_12)

data_21 = {'ID': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8'],
        'volume': [38, 9, 17, 9, 19, 38, 16, 24],
        'earning': [21790, 22215, 21000, 19135, 22435, 23630, 19855, 21140],
         'holding':[6,0,2,42,9,67,2,0],
         'session':21*np.ones(8,dtype=int)}
df_21 = pd.DataFrame(data_21)

data_22 = {'ID': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8'],
        'volume': [24, 41, 15, 11, 17, 11, 68, 4],
        'earning': [30305, 15450, 28050, 11535, 26870, 18150, 10660, 23780],
         'holding':[4, 18, 1, 42, 0, 18, 42, 3],
         'session':22*np.ones(8,dtype=int)}
df_22 = pd.DataFrame(data_22)

data_23 = {'ID': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8'],
        'volume': [12, 32, 28, 23, 86, 9, 22, 67],
        'earning': [26220, 34050, 26330, 29925, 33590, 26865, 28400, 23420],
         'holding':[36, 23, 2, 5, 6, 14, 42, 0],
         'session':23*np.ones(8,dtype=int)}
df_23 = pd.DataFrame(data_23)

data_24 = {'ID': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8'],
        'volume': [63, 5, 16, 34, 30, 62, 48, 39],
        'earning': [53490, 23645, 31970, 15870, 25110, 21780, 6440, 24895],
         'holding':[4, 22, 18, 7, 6, 0, 65, 6],
         'session':24*np.ones(8,dtype=int)}
df_24 = pd.DataFrame(data_24)
#df_1

# list_of_dfs = [df_1,df_2,df_3,df_4,df_5,df_6,df_7,df_8,df_9,df_10,df_11,df_12]
list_of_dfs = [df_21,df_22,df_23,df_24]

for i in range(0,4):
    list_of_dfs[i].to_csv('save_earnings'+str(i+21)+'.csv') # Careful !!str(i+21) for second set of experiments!!


# =============================================================================
# PB: Add volume; collect from saveit_official * v2.csv
# =============================================================================

# for s in range(1,13):
for s in range(21,25):
    d_vol = pd.read_csv('saveit_official '+str(s)+' v2.csv')
    d_earnings = pd.read_csv('save_earnings'+str(s)+'.csv')
    # Market orders: by owner_y (> 0)
    # Corresponding limit orders: corresponding owner_x_x
    # volume in trade_y
    # check whether sum equals total trades (records with trade_x_x == 1 (volume in trade_y)
    # This ignores multi-unit orders, so volume=trades!
    df_vol=pd.DataFrame(d_vol)
    df_vol_1=df_vol[df_vol['owner_y'] >= 1] # Market orders
    df_vol_2=df_vol[df_vol['trade_x_x'] == 1] # Executed limit orders

    # Check whether number of trades is the same whether gotten from limit orders (vol_1) or market orders (vol_2)
    df_vol_1.shape[0] == df_vol_2.shape[0]

    num_m_orders = df_vol_1.groupby(['owner_y']).sum()
    num_m_orders['trade_y']
    # Check whether total number of trades adds up across participants
    num_m_orders['trade_y'].sum() == df_vol_1.shape[0]

    num_l_orders = df_vol_1.groupby(['owner_x_x']).sum()
    num_l_orders['trade_y']
    # Check whether total number of trades adds up across participants
    num_l_orders['trade_y'].sum() == df_vol_1.shape[0]

    num_orders = pd.concat([num_m_orders['trade_y'], num_l_orders['trade_y']], axis=1)
    tot_orders = num_orders.sum(axis=1)
    # Check whether we obtain twice the volume
    tot_orders.sum() == 2 * df_vol_1.shape[0]

    d_earnings.index += 1 # Adjust index, update volume, and re-adjust index
    d_earnings['volume'] = tot_orders
    d_earnings.to_csv('save_earnings'+str(s)+'.csv')


#np.sum(df_5['volume'])/2
# =============================================================================
# Plot the final position in one plot
# =============================================================================

# PB Get back dataframes needed for plots from saved csv files
df_1 = pd.read_csv('save_earnings'+str(1)+'.csv')
df_2 = pd.read_csv('save_earnings'+str(2)+'.csv')
df_3 = pd.read_csv('save_earnings'+str(3)+'.csv')
df_4 = pd.read_csv('save_earnings'+str(4)+'.csv')
df_5 = pd.read_csv('save_earnings'+str(5)+'.csv')
df_6 = pd.read_csv('save_earnings'+str(6)+'.csv')
df_7 = pd.read_csv('save_earnings'+str(7)+'.csv')
df_8 = pd.read_csv('save_earnings'+str(8)+'.csv')
df_9 = pd.read_csv('save_earnings'+str(9)+'.csv')
df_10 = pd.read_csv('save_earnings'+str(10)+'.csv')
df_11 = pd.read_csv('save_earnings'+str(11)+'.csv')
df_12 = pd.read_csv('save_earnings'+str(12)+'.csv')

df_21 = pd.read_csv('save_earnings'+str(21)+'.csv')
df_22 = pd.read_csv('save_earnings'+str(22)+'.csv')
df_23 = pd.read_csv('save_earnings'+str(23)+'.csv')
df_24 = pd.read_csv('save_earnings'+str(24)+'.csv')


concate = pd.concat([df_1,df_2,df_3,df_4,df_5,df_6,df_7,df_8,df_9,df_10,df_11,df_12,df_21,df_22,df_23,df_24])
concate.to_csv("All_Earnings.csv")
# This can be saved in one file, to have all earnings, needed in analysis of cointegration/Granger causality
# data (see Matlab Life Scripts)
concate_1 = pd.concat([df_1,df_2,df_3,df_8,df_9,df_11,df_21,df_22,df_23,df_24])
concate_2 = pd.concat([df_4,df_5,df_6,df_7,df_10,df_12])
ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

# performance

plt.scatter(concate_1['session'],concate_1['earning']/100,marker='.')
plt.scatter(concate_2['session'],concate_2['earning']/100,marker='.')
plt.title("Earnings (Dashed Horizontal Line = Expected Earnings)")
plt.hlines(y=250, xmin=1, xmax=24,colors='k', linestyles='dashed', label='Average Earning')
plt.ylabel('Experimental Dollars')
plt.xlabel('Session Number')
plt.xticks()
plt.show()
plt.savefig('Earnings.eps')


# Total volume traded
ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.scatter(concate_1['session'],concate_1['volume'],marker='.')
plt.scatter(concate_2['session'],concate_2['volume'],marker='.')
plt.title("Volume Traded Per Participant")
#plt.hlines(y=np.mean(concate['volume']), xmin=1, xmax=5,colors='k', linestyles='dashed', label='Average Earning')
plt.ylabel('Number of Trades')
plt.xlabel('Session Number')
plt.xticks()
plt.show()
plt.savefig('Volume.eps')

#Final holding
ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.scatter(concate_1['session'],concate_1['holding'],marker='.')
plt.scatter(concate_2['session'],concate_2['holding'],marker='.')
plt.title("Final holdings of Asset Per Participant")
#plt.hlines(y=np.mean(concate['holding']), xmin=1, xmax=5,colors='k', linestyles='dashed', label='Average Earning')
plt.ylabel('Units of Asset')
plt.xlabel('Session Number')
plt.xticks()
plt.show()
plt.savefig('Final_Holdings.eps')

concate['volume'].describe()
df_1['earning'].describe()
df_2['earning'].describe()
df_3['earning'].describe()
df_4['earning'].describe()
df_5['earning'].describe()


################################################################################
# PART 3; READ SCR DATA, MERGE WITH MARKETS DATA AND ANALYZE CO-INTEGRATION
# Read earnings and split cointegration results into earrnings terciles and plot
# GENERATE FIG 3
################################################################################

#
# SCR DATA
#

# Time adjustment (order data lead ECG/SCR data by ... second, sessions 1...24):
time_adjust = [0, 0, 0, 0, 1, 25, 11, 20, 13, 12, 38, 14, 0, 0, 0, 0, 0, 0, 0, 0, 8, 5, 4, 10]
# pick your sessions to be analyzed:
pick_sessions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 21, 22, 23, 24]
# This is when looking at a subset:
pick_sessions = [21, 22, 23, 24]

time_format = '%H:%M:%S.%f'
for s in pick_sessions:
    for p in range(1, 9):
        if s == 1 and p == 3:  # Missing data
            continue
        if s == 2 and p == 3:  # Missing data
            continue
        if s == 6 and p == 7:  # Missing data
            continue
        R = {}
        # PB Note that sessions 21...24 are called differently, and need a new column "time in sec"
        if s < 20:
            R = pd.read_csv('session' + str(s) + '_sub' + str(p) + '.csv')
        else:
            ss = s - 20
            R = pd.read_csv('sessionEX' + str(ss) + 'M' + str(p) + '.csv')

        df_pm = pd.DataFrame(R)
        # add new column "time in sec" to data for sessions 21...24 (PB)
        if s > 20:
            times = df_pm['Hh:mm:ss']
            # The following code is only important (and time consuming!!) for:
            # (s,p) = (21,8), (22,6), (23, 3), (24, 3), (24, 4), (24, 8)
            # This code picks up very rare and slight variations on time recording in the Siemens data (no sec, no ms)
            for nmbr in range(len(times)):
                t = times[nmbr]
                if len(t) == 8:
                    times[nmbr] = t + '.000'  # PB BTW this is bad code since python changes not only the copy (t) but
                    # also the original times[nmbr], in a strange application of co-tangling (here harmless)
                elif len(t) == 5:
                    times[nmbr] = t + ':00.000'
            base_time = dt.strptime(times[0], time_format)
            seconds = []
            for t in times:
                seconds.append((dt.strptime(t, time_format) - base_time).total_seconds())

            seconds = np.array(seconds)
            seconds = seconds.astype(int)
            df_pm["time in sec"] = seconds

        df_pm = df_pm.groupby('time in sec', as_index=False)['MicroSiemens_LevelChange'].mean()

        df_pm.to_csv('GSR in sec session ' + str(s) + str(p) + '.csv')

# PB Here then goes the important merging of SCR and market data. 

for s in pick_sessions:
    for p in range(1, 9):
        if s == 1 and p == 3:  # Missing data
            continue
        if s == 2 and p == 3:  # Missing data
            continue
        if s == 6 and p == 7:  # Missing data
            continue
        d = []
        d = pd.read_csv('saveit_official ' + str(s) + ' v2.csv')
        df = pd.DataFrame(d)
        # PB: deprecated code: df = df.convert_objects(convert_numeric=True)
        cols = df.columns[df.dtypes.eq('object')]
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

        df_pm = pd.read_csv('GSR in sec session ' + str(s) + str(p) + '.csv')
        df['mispricing_' + str(p) + '_v2'].fillna(method='bfill', inplace=True)
        df['mispricing_' + str(p) + '_v2'] = df['mispricing_' + str(p) + '_v2'].apply(pd.to_numeric)
        df['mispricing'].fillna(method='bfill', inplace=True)
        # df['spread']=df['spread'].replace(np.NaN,'xx')
        # df['ask']=df['ask'].replace(np.NaN,'xx')
        # df['bid']=df['bid'].replace(np.NaN,'xx')
        df['time secs'] = df['time secs'].astype(int)
        # PB added this code to make order/trade data synchronous
        df['time secs'] = df['time secs'] + time_adjust[s - 1]
        # PB Following line is important: merge trading data ('time in sec') with SCR data ('time secs')
        # PB Made sure time is kept (from order data)  by using OUTER join after dropping duplicates in order data
        # Originally:
        # merge_pm = pd.merge_ordered(df_pm, df, left_on='time', right_on='time secs', how='outer')
        # First drop duplicates (same time in seconds) in order data, keep only first (least recent) record
        df = df.drop_duplicates(subset=['time secs'], keep='last')
        merge_pm = pd.merge_ordered(df_pm, df, left_on='time in sec', right_on='time secs', how='outer')
        merge_pm.fillna(method='ffill', inplace=True)
        merge_pm.rename(columns={'MicroSiemens_LevelChange': 'SCL_M'}, inplace=True)
        merge_pm.rename(columns={'s' + str(p) + ' holding_v2': 'Holding'}, inplace=True)
        merge_pm.rename(columns={'s' + str(p) + ' cash_v2': 'Cash'}, inplace=True)
        merge_pm.rename(columns={'Asset Value_market_v2_S' + str(p): 'Asset Value'}, inplace=True)
        merge_pm.rename(columns={'BUY:S' + str(p): 'Buy'}, inplace=True)
        merge_pm.rename(columns={'SELL:S' + str(p): 'Sell'}, inplace=True)
        merge_pm.rename(columns={'mispricing_' + str(p) + '_v2': 'Ind_mis'}, inplace=True)
        # PB Made sure time is kept
        coint_M = merge_pm[
            ['time in sec', 'SCL_M', 'Holding', 'Cash', 'Asset Value', 'Best Price', 'Ind_mis', 'mispricing', 'bid',
             'ask', 'spread']]
        # coint_M['ask']=coint_M['ask'].replace('xx',np.NaN)
        # coint_M['bid']=coint_M['bid'].replace('xx',np.NaN)
        # coint_M['spread']=coint_M['spread'].replace('xx',np.NaN)
        coint_M.to_csv('coint' + str(s) + '_s' + str(p) + '.csv')

# =============================================================================
# Cointegration between "variable" (market-relevant variable of interest) and SCR
# =============================================================================

# First the method, then the method calls.

def cointegration(variable):
    # table = np.array([np.arange(38)]*1).T PB: This appears to never be used (38 = # rows in output df)
    # table = pd.DataFrame(table)
    name = []
    # param = []
    # pvalue = []
    # rsquared = []
    # param_temp = []
    pvalue_temp = []
    adf_scr = []
    adf = []
    coint_valid = []
    ###################
    for s in pick_sessions:
        for p in range(1, 9):
            if s == 1 and p == 3:
                continue
            if s == 2 and p == 3:
                continue
            if s == 6 and p == 7:  # Missing data
                continue
            # read data
            R = pd.read_csv('coint' + str(s) + '_s' + str(p) + '.csv')
            data = pd.DataFrame(R)
            temp = []
            temp.append(data[variable][0])
            for i in range(1, len(data)):
                if data[variable][i] != data[variable][i - 1]:
                    temp.append(data[variable][i])
                if data[variable][i] == data[variable][i - 1]:
                    temp.append('xx')
            data['temp'] = temp
            scr = []
            v = []  # Here go the variable investigated on co-integration with
            for i in range(len(data)):
                datapoint = data['temp'][i]
                if datapoint != 'xx' and not np.isnan(datapoint):  # PB Added "NaN" condition to ensure stats work
                    v.append(data['temp'][i])
                    scr.append(data['SCL_M'][i])

            model = ts.coint(v, scr)  # (H0 = there is NO cointegration)
            results = model
            result1 = adfuller(v)  # unit root test on "variable" (H0 = there IS a unit root)
            adf.append(result1[1])
            result2 = adfuller(scr)  # unit root test for scr (H0 = there IS a unit root)
            adf_scr.append(result2[1])
            if result1[1] < 0.01 and result2[1] < 0.01:
                coint_valid.append('notok')
            else:
                coint_valid.append('ok')
            name.append('s' + str(s) + 'p' + str(p))
            pvalue_temp.append(results[1])  # cointegration p value

    adf = ['%.4f' % elem for elem in adf]
    adf_scr = ['%.4f' % elem for elem in adf_scr]
    pvalue_temp = ['%.4f' % elem for elem in pvalue_temp]
    df = pd.DataFrame({'participants': name, 'D.fuller': adf, 'D.fuller scr': adf_scr, \
                       'p-value coint': pvalue_temp, 'coint_valid': coint_valid})
    return df


# =============================================================================
# Run cointegration test between SCR and Market variables
# =============================================================================
dfH = cointegration('Holding')
dfC = cointegration('Cash')
dfAV = cointegration('Asset Value')
dfBP = cointegration('Best Price')
dfIM = cointegration('Ind_mis')
dfM = cointegration('mispricing')
dfB = cointegration('bid')
dfA = cointegration('ask')
dfS = cointegration('spread')  

# Provisional saving to check calculations (PB):
dfAV.to_csv("CointSCR_AssetValueHolding.csv")
dfC.to_csv("CointSCR_CashHolding.csv")
dfS.to_csv("CointSCR_BASpread.csv")

###COPY THE P-VALUE INTO AN EXCEL SPREADSHEET P-value_aggregate.csv AND NAME VARIABLES:
# Market Mispricing
# Individual  Mispricing
# Total Vol PB ??
# Asset value
# Holding
# Cash
# BID 
# ASK
# Spread # PB added

df_all = pd.concat([dfM['p-value coint'], dfIM['p-value coint'], dfAV['p-value coint'], dfH['p-value coint']], axis=1)
df_all = pd.concat([df_all, dfC['p-value coint'], dfB['p-value coint'], dfA['p-value coint'], dfS['p-value coint']],
                   axis=1)
df_all.columns = ['Mispricing', 'Individual Mispricing', 'Asset value', 'Holding', 'Cash', 'BID', 'ASK', 'Spread']

df_all.to_csv("P-value_aggregate.csv")

# Save all 
df_all = pd.concat([dfM, dfIM, dfAV, dfH], axis=1)
df_all = pd.concat([df_all, dfC, dfB, dfA, dfS], axis=1)
df_all.to_csv("CointSCR_aggregate.csv")

# Plot individual series (SCR, variable), save data for extra stats analysis outside python program
variable = 'spread'
s = 7
p = 5

R = pd.read_csv('coint' + str(s) + '_s' + str(p) + '.csv')
data = pd.DataFrame(R)
temp = []
temp.append(data[variable][0])
for i in range(1, len(data)):
    if data[variable][i] != data[variable][i - 1]:
        temp.append(data[variable][i])
    if data[variable][i] == data[variable][i - 1]:
        temp.append('xx')
data['temp'] = temp
scr = []
v = []  # Here go the variable investigated on co-integration with
for i in range(len(data)):
    datapoint = data['temp'][i]
    if datapoint != 'xx' and not np.isnan(datapoint):  # PB Added "NaN" condition to ensure stats work
        v.append(data['temp'][i])
        scr.append(data['SCL_M'][i])

dfv = pd.DataFrame(v)
dfscr = pd.DataFrame(scr)
forplot = pd.concat([dfv, dfscr], axis=1)
forplot.columns = [variable, 'scr']
forplot.to_csv('scrvarForPlot.csv')


################################################################################
# PART 4; READ HR DATA, MERGE WITH MARKETS DATA AND ANALYZE VAR/Granger Causality
################################################################################


# =============================================================================

#                         HR data against Market Data

# Synchronization takes into account putative HR response delays of 0s, 5s, ...
# See "time_adjust," which adjusts both for recorded time differences between
# ECG and markets data (some sessions) AND putative HR response delay
# (Line 1147: time_adjust = time_adjust - <?>*np.ones(24) where <?> is putative
# HR response delay, i.e. 0, 5, 10, ...
# Keep track of putative HR response delay to correctly name analysis output file
# (E.g., "CograngerHR_aggregate1to24s50.csv", which means Granger causality tests
# for sessins 1 to 24 with 10s putative HR response delay)

# =============================================================================

# PB: Add sec in hr data; delete 1/2 second observations
# Sec are added to ensure synchrony with SCR data

s1 = 4*60+52
s2 = 4*60+58
s3 = 4*60+39
s4 = 4*60+44
s5 = 5*60+25
s6 = 0
s7 = 0
s8 = 0
s9 = 0
s10 = 0
s11 = 0
s12 = 0

s21 = 0
s22 = 0
s23 = 0
s24 = 0

# Time adjustment (order data lead ECG/SCR data by ... second, sessions 1...24):
time_adjust = [0, 0, 0, 0, 1, 25, 11, 20, 13, 12, 38, 14, 0, 0, 0, 0, 0, 0, 0, 0, 8, 5, 4, 10]
# time_adjust = np.zeros(24)
# The following adjusts for slow heartrate response (up to 30s):
time_adjust = time_adjust - 10*np.ones(24)

def HRV_add_time(s,ss):
    d={}
    df= {}
    for p in range(1,9):
        if p == 3 and s == 1: # Missing data
            continue
        if s == 2 and p == 3:  # Missing data
            continue
        if s == 6 and p == 5:  # Missing data
            continue
        if s == 11 and (p == 1 or p == 2):  # Missing data
            continue
        dp=pd.read_csv('S' + str(s) + '_R' + str(p) + '.csv') # PB I added Session indicator to file names
        dfp=pd.DataFrame(dp)
        temp = []
        for i in range(len(dp)):
            if (-int(ss))+0.5*i < 0 or np.mod(i,2) > 0: # PB I drop records with negative time
                dfp = dfp.drop(dp.index[i])
            else:
                temp.append((-int(ss))+0.5*i)
        dfp['time']=temp
        dfp.to_csv('Session '+str(s)+' P'+str(p)+'.csv')

HRV_add_time(1,s1)
HRV_add_time(2,s2)
HRV_add_time(3,s3)
HRV_add_time(4,s4)
HRV_add_time(5,s5)
HRV_add_time(6,s6)
HRV_add_time(7,s7)
HRV_add_time(8,s8)
HRV_add_time(9,s9)
HRV_add_time(10,s10)
HRV_add_time(11,s11)
HRV_add_time(12,s12)

HRV_add_time(21,s21)
HRV_add_time(22,s22)
HRV_add_time(23,s23)
HRV_add_time(24,s24)

# =============================================================================
#
#                Merge market data with HRV data
#
# PB changed time to per second only and if there is an order, i.e.,
#           time change = max(second, order time)
# So record only keeps last order in a second if multiple orders within a second
#
# =============================================================================

#for s in range(1, 13):
for s in range(1, 25):
    for p in range(1, 9):
        if s == 1 and p == 3:  # Missing data
            continue
        if s == 2 and p == 3:  # Missing data
            continue
        if s == 6 and p == 5:  # Missing data
            continue
        if s == 11 and (p == 1 or p == 2):  # Missing data
            continue
        if s >= 13 and s < 21:
            continue
        d = []
        d = pd.read_csv('saveit_official ' + str(s) + ' v2.csv')
        df = pd.DataFrame(d)
        # PB: deprecated code: df = df.convert_objects(convert_numeric=True)
        cols = df.columns[df.dtypes.eq('object')]
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
        R = {}
        R = pd.read_csv('Session '+str(s)+' P'+str(p)+'.csv')
        df_pm = pd.DataFrame(R)

        df['mispricing_' + str(p) + '_v2'].fillna(method='bfill', inplace=True)
        df['mispricing_' + str(p) + '_v2'] = df['mispricing_' + str(p) + '_v2'].apply(pd.to_numeric)
        df.rename(columns={'mispricing_' + str(p) + '_v2': 'Ind_mis'}, inplace=True)
        df['bid'].fillna(method='bfill', inplace=True)
        df['bid'] = df['bid'].apply(pd.to_numeric)
        df['ask'].fillna(method='bfill', inplace=True)
        df['ask'] = df['ask'].apply(pd.to_numeric)
        # Try RELATIVE mispricing to avoid nonstationarity
        df.eval('R_Ind_mis = Ind_mis/bid*(bid>0)', inplace=True)
        # df['R_Ind_mis'] = df['R_Ind_mis'],replace(np.NaN,0)
        df['mispricing'].fillna(method='bfill', inplace=True)
        #df['spread'] = df['spread'].replace(np.NaN, 'xx')
        #df['ask'] = df['ask'].replace(np.NaN, 'xx')
        #df['bid'] = df['bid'].replace(np.NaN, 'xx')
        df['time secs'] = df['time secs'].astype(int)
        # PB added this code to make order/trade data synchronous
        df['time secs'] = df['time secs'] + time_adjust[s-1]
        # PB Following line is important: merge trading data ('time in sec') with HR data ('time secs')
        # PB Made sure time is kept (from order data)  by using OUTER join after dropping duplicates in order data
        # Originally:
        # merge_pm = pd.merge_ordered(df_pm, df, left_on='time', right_on='time secs', how='outer')
        # First drop duplicates (same time in seconds) in order data, keep only first (least recent) record
        df = df.drop_duplicates(subset=['time secs'], keep='last')
        merge_pm = pd.merge_ordered(df_pm, df, left_on='time', right_on='time secs', how='outer')
        merge_pm.fillna(method='ffill', inplace=True)
        merge_pm.rename(columns={'hr': 'heartrate'}, inplace=True)
        merge_pm.rename(columns={'s' + str(p) + ' holding_v2': 'Holding'}, inplace=True)
        merge_pm.rename(columns={'s' + str(p) + ' cash_v2': 'Cash'}, inplace=True)
        merge_pm.rename(columns={'Asset Value_market_v2_S' + str(p): 'Asset Value'}, inplace=True)
        merge_pm.rename(columns={'BUY:S' + str(p): 'Buy'}, inplace=True)
        merge_pm.rename(columns={'SELL:S' + str(p): 'Sell'}, inplace=True)
        # merge_pm.rename(columns={'Imispricing_' + str(p) + '_v2': 'Ind_mis'}, inplace=True)
        # PB Made sure time is kept
        # PB Make sure time change = second
        coint_M = merge_pm[
            ['time', 'heartrate', 'Holding', 'Cash', 'Asset Value', 'Best Price', 'Ind_mis', 'R_Ind_mis', 'mispricing', 'bid',
             'ask', 'spread']]
        #coint_M['ask'] = coint_M['ask'].replace('xx', np.nan)
        #coint_M['bid'] = coint_M['bid'].replace('xx', np.nan)
        #coint_M['spread'] = coint_M['spread'].replace('xx', np.nan)
        coint_M.to_csv('HRcoint' + str(s) + '_s' + str(p) + '.csv') # PB Note Prefix HR to distinguish from SCR data

# ====================================================================================================
#
# ++++++++++++++++++ NOW analyze with VAR/Granger causality rather than cointegration +++++++++++++++
#
# Make sure results are written to file with name that clearly indicates which sessions are dealt with
# and which putative heart rate response delay has been applied (line 1425)
#
# ====================================================================================================

# Granger causality analysis

def cogranger(variable):
    # table = np.array([np.arange(38)]*1).T PB: This appears to never be used (38 = # rows in output df)
    # table = pd.DataFrame(table)
    HR_MA = 0 # LEGACY CODE from when MA of HR responses were used; HR_MA = 0 means: no MA.
    # Parameter HR_MA was used to compute predictor as average over HR_MA PAST observations
    # (if < 0) or HR_MA future observations (if > 0)
    # Note: average will be over HR_MA+1 observations
    name = []
    # param = []
    # pvalue = []
    # rsquared = []
    # param_temp = []
    gc_hr = []
    gc_v = []
    gcc_hr = []
    gcc_v = []
    # coint_valid = []
    ###################
    #for s in range(1, 13):
    for s in range(3, 4):
        for p in range(3, 4):
            print(s,p)
            if s == 1 and p == 3:
                continue
            if s == 2 and p == 3:
                continue
            if s == 6 and p == 5:  # Missing data
                continue
            if s == 11 and (p == 1 or p == 2):  # Missing data
                continue
            if s >= 13 and s < 21:
                continue
            # read data
            R = pd.read_csv('HRcoint' + str(s) + '_s' + str(p) + '.csv')
            data = pd.DataFrame(R)
            hr = [] # Here goes heart rate
            v = []  # Here goes the variable investigated on Granger causality with hr
            # PB: When heart rate data are adjusted for slow response times, need to eliminate records with no time
            # (These records don't have a heart rate recorded since "time" is taken from heart rate data)
            tm = [] # Here goes time
            check_start = 0
            while np.isnan(data['time'][check_start]):
                check_start += 1
            # The following code is like original, where records were included only if there was a change
            # There are always heartrate data because of the check_start, but not always "variable" entries so skip those
            for i in range(check_start, len(data)):
                datapoint = data[variable][i]
                if i < -HR_MA:
                    continue
                elif i == 0:  # won't happen for HR_MA < 0
                    if not np.isnan(datapoint):  # PB Added "NaN" condition to ensure stats work
                        v.append(data[variable][i])
                        avg_hr = np.mean(data['heartrate'][i:i+HR_MA+1])  # HR_MA >= 0 necessarily because i >= -HR_MA
                        # Note typical python problem : if wanting elements i, i+1, ..., i+HR_MA need i:i+HR_MA+1
                        hr.append(avg_hr)
                        tm.append(data['time'][i])
                elif i <= min(len(data)-HR_MA,len(data)):
                    prevdatapoint = data[variable][i-1] # This requires check_start > 0
                    if datapoint != prevdatapoint and not np.isnan(datapoint):  # PB Added "NaN" condition to ensure stats work
                        if HR_MA < 0:
                            avg_hr = np.mean(data['heartrate'][i+HR_MA:i+1])
                            # Note typical python problem : if wanting elements i+HR_MA,...,i need i+HR_MA:i+1 (HR_MA < 0)
                        else:
                            avg_hr = np.mean(data['heartrate'][i:i+HR_MA+1])
                        v.append(data[variable][i])
                        hr.append(avg_hr)
                        tm.append(data['time'][i])

            dfv = pd.DataFrame(v)
            dfhr = pd.DataFrame(hr)
            dftm = pd.DataFrame(tm)
            data = pd.concat([dfv, dfhr], axis=1)
            data.columns = [variable, 'heartrate']

            # Plot data (see also Fig 4, though this is reproduced using matlab code; see Figures.mlx
            dfv.columns = ['v']
            dfv = dfv - dfv.mean(axis=0)
            dfv = dfv/dfv.std(axis=0)
            dfhr.columns = ['h']
            dfhr = dfhr - dfhr.mean(axis=0)
            dfhr = dfhr / dfhr.std(axis=0)
            datatoplot = pd.concat([dftm, dfv, dfhr], axis=1)
            datatoplot.columns = ['time', variable, 'heartrate']
            plt.plot(datatoplot['time'], datatoplot['heartrate'], '.r-', label='HR')
            plt.plot(datatoplot['time'], datatoplot[variable], '.b-', label='B/A Spread')
            plt.title("Subject 3 in Session 3 (Spread CAUSES HR, p=0.01)")
            plt.ylabel('Standardized Outcomes')
            plt.xlabel('Time (secs since session begin)')
            plt.legend()
            plt.xticks()
            plt.show()
            epsname = 'HeartrateBASpread_s' + str(s) + '_p' + str(p)
            plt.savefig(epsname)
            # plt.clf()
            datatoplot.to_csv('HeartrateBASpread_s' + str(s) + '_p' + str(p) + '.csv')

            model = VAR(data)
            result = model.fit()

            # plt.acorr(result.resid[variable])
            # plt.show()
            # plt.acorr(result.resid['scr'])
            # plt.show()

            results = result.test_causality(variable, 'heartrate')  # Arguments: caused, causing
            gcc_hr.append(results.test_statistic)
            gc_hr.append(results.pvalue)
            results = result.test_causality('heartrate', variable)
            gcc_v.append(results.test_statistic)
            gc_v.append(results.pvalue)
            name.append('s' + str(s) + 'p' + str(p))

    gc_v = ['%.4f' % elem for elem in gc_v]
    gc_hr = ['%.4f' % elem for elem in gc_hr]
    gcc_v = ['%.4f' % elem for elem in gcc_v]
    gcc_hr = ['%.4f' % elem for elem in gcc_hr]
    df = pd.DataFrame({'participants': name, 'GC heartrate': gc_hr, 'GCC heartrate': gcc_hr, 'GC v': gc_v, 'GCC v': gcc_v})
    return df


# =============================================================================
# Run Granger causality test between HR and Market variables
# =============================================================================
dfH = cogranger('Holding')
# dfC = cogranger('Cash')
# dfAV = cogranger('Asset Value')
# dfBP = cogranger('Best Price')
dfIMR = cogranger('R_Ind_mis')
dfIMA = cogranger('Ind_mis')
# dfM = cogranger('mispricing')
# dfB = cogranger('bid')
# dfA = cogranger('ask')
dfS = cogranger('spread')  # PB Don't we need spread as well? I added it

###COPY THE P-VALUE INTO AN EXCEL SPREADSHEET P-value_aggregate.xlsx AND TABLE THE TITLE AS:
# Market Mispricing
# Individual  Mispricing
# Total Vol PB ??
# Asset value
# Holding
# Cash
# BID
# ASK
# Spread # PB added

# Save all
df_all = pd.concat([dfIMA,dfIMR, dfH,dfS],axis=1)
# df_all = pd.concat([df_all,dfC,dfB,dfA,dfS],axis=1)
# Make sure filename reflects which sessions are being evaluated as well as HR response delay
df_all.to_csv("CograngerHR_aggregate1to24s10.csv")

# +++++++++++++++++++++++++++++++++++ END PB CODE +++++++++++++++++++++++++++++
