# -*- coding: utf-8 -*-
"""ete011 @Enrico Tedeschi - UiT
Observing the Bitcoin blockchain in real time. The system will retreive portion of the Bitcoin blockchain,
do data analysis, generating models and plotting the results.
Usage: observ.py -t number
    -h | --help         : usage
    -i                  : gives info of the blockchain in the file .txt
    -e number           : add/retrieve <number> blocks from the last one created if the file doesn't exist.
    -p                  : plot data
    -t number           : retrieve <number> blocks (the most recent ones) but saves only the transactions
    -d                  : retrieve all transactions and save them in a Panda DataSet

"""

import sys, getopt
import numpy as np
import string
import re
import os.path
import matplotlib.pyplot as plt
import datetime
import time
import statsmodels.api as sm
import matplotlib.lines as mlines
import matplotlib.axis as ax
import io
import urllib2
import matplotlib.patches as mpatches
import json
import ast
import urllib
import math
import matplotlib as mpl
import matplotlib.ticker
import calendar
import bisect
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import csv
import javabridge
import subprocess as sub

from scipy.stats import spearmanr
from matplotlib.colors import LogNorm
from blockchain import blockexplorer
from time import sleep
from forex_python.converter import CurrencyRates
from forex_python.bitcoin import BtcConverter
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from scipy import stats
from scipy.stats import norm
from docopt import docopt
from matplotlib.ticker import FormatStrFormatter
from shutil import copyfile
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import linear_model
from sklearn.metrics import *

# ------ GLOBAL ------
global txs_dataset
txs_dataset = "transaction_dataframe.tsv"

global file_name_old
file_name = "blockchain_new.txt"

global file_name
file_name = "blockchain_for_txs.txt"

global file_info_new
file_info_new = "file_info_new.txt"

global file_tx
file_tx = "transactions_new.txt"

global latest_block_url
latest_block_url = "https://blockchain.info/latestblock"

global unconfirmed_txs_url
unconfirmed_txs_url = "https://blockchain.info/unconfirmed-transactions?format=json"

global block_hash_url
block_hash_url = "https://blockchain.info/rawblock/"

global b
b = BtcConverter()

global latest_price
# latest_price = b.get_latest_price('USD')
latest_price = 2000.0

# --------------------

def main(argv):
    try:
        global plot_number
        plot_number = 0
        args_list = sys.argv
        args_size = len(sys.argv)
        earliest_hash = get_earliest_hash()
        start_v = None
        end_v = None

        opts, args = getopt.getopt(argv, "hipdt:e:")
        valid_args = False

        for opt, arg in opts:
            if (opt == "-e"):  # append blocks at the end
                print ("Appending at the end " + arg + " blocks.")
                number_of_blocks = int(arg)
                # retrieve blocks and save them
                if ((earliest_hash == False) or (earliest_hash == "")):
                    # file doesn't exist. Retrieve the last block
                    latest_block = get_json_request(latest_block_url)
                    earliest_hash = latest_block['hash']
                get_blockchain(number_of_blocks, earliest_hash)
                valid_args = True
            if (opt == "-t"):   # save transactions fee
                height_to_start = 470000
                b_array = get_json_request(
                    "https://blockchain.info/block-height/" + str(height_to_start) + "?format=json")
                #latest_block = get_json_request(latest_block_url)
                blocks = b_array['blocks']
                latest_block = blocks[0]
                earliest_hash = latest_block['hash']
                number_of_blocks = int(arg)
                save_transactions(number_of_blocks, earliest_hash)
                valid_args = True
            if (opt == "-i"):   # blockchain info
                print blockchain_info()
                valid_args = True
            if (opt == "-p"):
                plot()
                valid_args = True
            if (opt == "-d"):
                # fetch transactions
                fetch_txs()
                # create dataset
                read_txs_file()
                # plot dataset
                valid_args = True
            if (opt == "-h"):  # usage
                print (__doc__)
                valid_args = True
        if (valid_args == False):
            print (__doc__)
    except getopt.GetoptError:
        print (__doc__)
        sys.exit(2)



def fetch_txs():
    if (os.path.isfile(txs_dataset)):
        # file already exists add data to the dataset
        # get the latest block hash

        df = pd.DataFrame.from_csv(txs_dataset, sep='\t')
        hash_list = df['B_h'].values

        # retrieve 100 blocks earlier
        height_list = df['B_he'].values
        last_block = height_list[-1]
        last_block = int(last_block) - 100


        b_array = get_json_request("https://blockchain.info/block-height/" + str(last_block) + "?format=json")
        blocks = b_array['blocks']
        b = blocks[0]

        block_hash = b['hash']

        # remove the blockchain file
        if (os.path.isfile(file_name)):
            os.remove(file_name)

        # remove transactions file
        if (os.path.isfile(file_tx)):
            os.remove(file_tx)

        get_blockchain(5, block_hash)
        save_transactions(5, block_hash)

    else:
        # file doesn't exist
        # retrieve the last block hash
        latest_block = get_json_request(latest_block_url)
        block_hash = latest_block['hash']
        get_blockchain(5, block_hash)
        save_transactions(5, block_hash)



def read_txs_file():
    """
    read the txs file and generates the dataset
    :return:
    """
    epoch_list = []
    fee_list = []
    size_list = []
    approval_time_list = []
    input = []
    output = []
    hash_tx = []

    # the file transactions.txt contains the transactions in each block and the epoch for that block at the end
    if (os.path.isfile(file_tx)):
        with io.FileIO(file_tx, "r") as file:
            file.seek(0)
            txs = file.read()

        list_txs = txs.split("\n")
        list_txs.pop()

        # -------- PROGRESS BAR -----------
        index_progress_bar = 0
        prefix = 'Reading ' + file_tx + ':'
        progressBar(index_progress_bar, 'Reading ' + file_tx + ':', (2 * len(list_txs)) + (len(epoch_list)))
        # ---------------------------------

        # delete the epoch from the list just retrieved
        i = 0
        for el in list_txs:
            epoch_list.append(list_txs[i + 1])
            list_txs.remove(list_txs[i + 1])
            i += 1
            # ---------- PROGRESS BAR -----------
            index_progress_bar += 1
            progressBar(index_progress_bar, prefix, (2 * len(list_txs)) + (len(epoch_list)))
            # -----------------------------------

        i = 0
        for t in list_txs:
            list_txs[i] = ast.literal_eval(t)
            i += 1
            # ---------- PROGRESS BAR -----------
            index_progress_bar += 1
            progressBar(index_progress_bar, prefix, (2 * len(list_txs)) + (len(epoch_list)))
            # -----------------------------------

        for i in range(len(epoch_list)):
            # ---------- PROGRESS BAR -----------
            index_progress_bar += 1
            progressBar(index_progress_bar, prefix, (2 * len(list_txs)) + (len(epoch_list)))
            # -----------------------------------

            list_txs[i].pop(0)  # remove the first transaction of each block since it is only the reward
            temp_input, temp_output, temp_fee_list, temp_size_list, temp_approval_time_list, temp_hash_tx = calculate_transactions_fee(
                list_txs[i], int(epoch_list[i]))
            input.extend(temp_input)
            output.extend(temp_output)
            fee_list.extend(temp_fee_list)
            size_list.extend(temp_size_list)
            approval_time_list.extend(temp_approval_time_list)
            hash_tx.extend(temp_hash_tx)

        # ---- CALCULATE % OF FEE ----
        f_percentile = []
        for f_in, f_ou in zip(input, output):
            percentile = 100 - (float(f_ou * 100) / float(f_in))
            f_percentile.append(percentile)
            # ----------------------------

        transactions_list = get_list_from_file('transactions')
        transactions_list[:] = [int(x) for x in transactions_list]
        transactions_list[:] = [x - 1 for x in transactions_list]

        # total_txs is the maximum index of the transactions saved in the dataframe
        total_txs = sum(transactions_list)


        indexes_list = []
        val = 0
        for x in transactions_list:
            val = x + val
            indexes_list.append(val)

        block_size = get_list_from_file('size')
        block_creation_time = get_list_from_file('creation_time')
        block_height = get_list_from_file('height')
        block_epoch = get_list_from_file('epoch')
        block_txs = get_list_from_file('transactions')
        block_hash = get_list_from_file('hash')


        b_s = []
        b_ct = []
        b_h = []
        b_ep = []
        b_t = []
        b_hash = []

        i = 0
        counter = 0
        for tx in input:
            if(i < indexes_list[counter]):
                b_s.append(block_size[counter])
                b_ct.append(block_creation_time[counter])
                b_h.append(block_height[counter])
                b_ep.append(block_epoch[counter])
                b_t.append(block_txs[counter])
                b_hash.append(block_hash[counter])
                i += 1
            else:
                counter += 1
                b_s.append(block_size[counter])
                b_ct.append(block_creation_time[counter])
                b_h.append(block_height[counter])
                b_ep.append(block_epoch[counter])
                b_t.append(block_txs[counter])
                b_hash.append(block_hash[counter])
                i += 1

        if (os.path.isfile(txs_dataset)):
            # file exists
            # read the old file:
            old_df = pd.DataFrame.from_csv(txs_dataset, sep='\t')

            # create the new one
            new_df = pd.DataFrame.from_items(
                [('t_ha', hash_tx), ('t_in', input), ('t_ou', output), ('t_f', fee_list), ('t_q', size_list),
                 ('t_%', f_percentile), ('t_l', approval_time_list),
                 ('Q', b_s), ('B_T', b_ct), ('B_he', b_h), ('B_ep', b_ep), ('B_t', b_t),
                 ('B_h', b_hash)])

            # merge old and new
            new_df = pd.concat([old_df, new_df])


        else:
            # file doesn't exist
            new_df = pd.DataFrame.from_items(
                [('t_ha', hash_tx), ('t_in', input), ('t_ou', output), ('t_f', fee_list), ('t_q', size_list),
                 ('t_%', f_percentile), ('t_l', approval_time_list),
                 ('Q', b_s), ('B_T', b_ct), ('B_he', b_h), ('B_ep', b_ep), ('B_t', b_t),
                 ('B_h', b_hash)])


        new_df.to_csv('transaction_dataframe.tsv', sep='\t')



def get_blockchain_growth():
    """

    :return: x, y lists. Where x is the epoch and y is the size in MB
    """
    growth = get_json_request("https://api.blockchain.info/charts/blocks-size?timespan=all&format=json")
    growth = growth['values']

    x = []
    y = []  # in MB

    for el in growth:
        x.append(el['x'])
        y.append(el['y'])

    return x, y

def calculate_percentage_txs_fee():
    """

    :return: array with the % of fee paid on the net transaction
    """
    # calculate % of fee considering the net output:
    df = pd.DataFrame.from_csv(txs_dataset, sep='\t')

    output = df['t_ou'].values
    fee = df['t_f'].values

    percentage_output = []

    for o, f in zip(output, fee):
        if (o == 0):
            o = 0.000001
        p = (float(f) * 100) / float(o)
        percentage_output.append(p)

    return percentage_output


def date_to_plot(epoch_list):
    """
    :param      epoch_list      list of epochs
    :return:    d               a list with the correct date to plot on labels
    """
    epoch_list[:] = [int(x) for x in epoch_list]
    epoch_list[:] = [epoch_datetime(x) for x in epoch_list]

    i = 0
    for date in epoch_list:
        date = time.strptime(date, "%d-%m-%Y %H:%M:%S")
        yy = date.tm_year
        mm = calendar.month_name[date.tm_mon][:3]
        dd = date.tm_mday
        epoch_list[i] = str(dd) + "-" + mm + "-" + str(yy)[-2:]
        i += 1
    d = epoch_list
    return d


def plot(miner=1):
    """
    plot the data retrieved
    :return: 
    
    """
    """epoch_list = get_list_from_file('epoch')

    epoch_list[:] = [float(x) for x in epoch_list]
    epoch_list.sort()
    start = epoch_datetime(epoch_list[0])
    end = epoch_datetime(epoch_list[-1])"""
    # ============= MINER'S PROFIT =================
    """
    plt.figure(0)
    axes = plt.gca()
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.xaxis.set_ticks_position('bottom')
    axes.yaxis.set_ticks_position('left')
    profit_list = []
    revenue_list = []
    cost_list = []

    #sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 2})

    epoch_list, fee_list, creation_time_list = get_lists_ordered("epoch", "fee", "creation_time")
    fee_list[:] = [float(x) for x in fee_list]
    fee_list[:] = [x / 100000000 for x in fee_list]  # in BTC
    creation_time_list[:] = [float(x) for x in creation_time_list]

    i = 0
    for x in creation_time_list:
        if (x <= 0):
            creation_time_list[i] = 1
        i += 1

    i = 0
    for x in fee_list:
        profit, revenue, cost = calculate_profit(fee_list[i], creation_time_list[i], miner)
        profit_list.append(profit)
        revenue_list.append(revenue)
        cost_list.append(cost)
        i += 1

    #sns.jointplot(np.asarray(creation_time_list), np.asarray(profit_list), xlim=(0.0, 4000.0), ylim=(0.0, 0.00007), stat_func = spearmanr, color="orange")
    #todo: PLOT fee, creation time and profit
    #plt.plot(creation_time_list, fee_list, "ro", label="$M$" + "\n" + start + "\n"+ end)
   
    together = zip(creation_time_list, profit_list)
    sorted_together = sorted(together)
    creation_time_list = [el[0] for el in sorted_together]
    profit_list = [el[1] for el in sorted_together]
    

    #plt.plot(creation_time_list, profit_list, "go", label=r"$\langle \Pi \rangle $" + "\n" + start + "\n"+ end, markevery=5)


    plt.plot(creation_time_list, revenue_list, "bo", label=r"$\langle V \rangle$")
    plt.plot(creation_time_list, cost_list, "ro", label=r"$\langle C \rangle$")

    plt.title("\nHashing rate: 100 MH/s.\nConsumption: 6.8 Watt.")

    # -- REGRESSION
    
    regression = []
    regression.append(r"$f_{\langle \Pi \rangle}(\mathcal{T})$")
    regression.append(37)
    regression.append(2)
    new_x, new_y, f = polynomial_interpolation(r"$f^{36}_{\langle \Pi \rangle}(\mathcal{T})$", creation_time_list, profit_list, regression[1])
    plt.plot(new_x, new_y, "r-", label=r"$f^{36}_{\langle \Pi \rangle}(\mathcal{T})$", lw=3)
    
    # -------------
    #axes.set_ylim([0, 0.00007])

    #axes.set_xlim([0, 4000])
    #axes.set_ylim([0, 6])

    plt.legend(loc="best")
    plt.xlabel("$\mathcal{T}$ (sec)")
    plt.ylabel("fee (BTC)")

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 18}

    matplotlib.rc('font', **font)

    plt.savefig('plot/miner_profit', bbox_inches='tight', dpi=500)
    print("plot miner_profit.png created")

    # --- ACCURACY ----
    real = profit_list
    together = zip(creation_time_list, real)
    sorted_together = sorted(together)

    creation_time_list = [x[0] for x in sorted_together]
    real = [x[1] for x in sorted_together]


    predicted = []
    for el in creation_time_list:
        x = el
        pred_y = f(x)
        predicted.append(pred_y)

    print mean_absolute_error(real, predicted)
    # -----------------

    # ===== FIND POLYNOM -- ACCURACY =====
    # test accuracy with more polynomial's degree

    # create 40 functions
    # remove value above 5000 sec
    # order lists according to creation time list
    
    real = profit_list

    together = zip(creation_time_list, real)
    sorted_together = sorted(together)

    creation_time_list = [x[0] for x in sorted_together]
    real = [x[1] for x in sorted_together]

    mean_absolute_error_list = []
    i = 1
    while i <= 40:
        new_x, new_y, f = polynomial_interpolation(regression[0], creation_time_list, real, i)
        predicted = []
        for el in creation_time_list:
            x = el
            pred_y = f(x)

            predicted.append(pred_y)
        # new_y are the predicted
        # predicted = new_y

        #calculate the mean absolute error
        mean_absolute_error_list.append(mean_absolute_error(real, predicted))
        i += 1

    best_value = min(mean_absolute_error_list)
    index = mean_absolute_error_list.index(best_value)

    print mean_absolute_error_list
    print "best mean: " + str(best_value) + "\n"
    print "polynom: " + str(index)
    
    # =========================================
    """
# ============= MINER'S PROFIT =================

# ============= GROWTH BLOCKCHAIN =================
    """
    plt.figure(2)
    axes = plt.gca()
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.xaxis.set_ticks_position('bottom')
    axes.yaxis.set_ticks_position('left')
    epoch_vals, x_vals, y_vals = get_lists_ordered("epoch", "creation_time", "size")

    epoch_vals[:] = [int(x) for x in epoch_vals]

    x_btc_info = []
    y_btc_info = []
    # get data from blockchain.info
    growth = get_json_request("https://api.blockchain.info/charts/blocks-size?format=json&timespan=all")
    growth_values = growth['values']
    for el in growth_values:
        bisect.insort_left(x_btc_info, int(el['x']))
        bisect.insort_left(y_btc_info, float(el['y']/1000))

    #plt.plot(x_btc_info, y_btc_info, "ro", label="growth from blockchain.info", markevery=10)

    dist = 10000000000000
    new_dist = 10000000000000
    i = 0
    for el in x_btc_info:
        new_dist = epoch_vals[0] - el
        if (abs(new_dist) < dist):
            dist = new_dist
            initial_size = y_btc_info[i]
        i += 1

    x_vals[:] = [float(x) for x in x_vals]
    x_vals[:] = [x / (60 * 60) for x in x_vals]  # in hours

    y_vals[:] = [float(x) for x in y_vals]
    y_vals[:] = [y / 1000000000 for y in y_vals]  # in GB

    x_vals = create_growing_time_list(x_vals)
    y_vals = create_growing_size_list(y_vals, initial_size=0)

    #plt.plot(x_vals, y_vals, "go", label="blockchain growth in between:\n" + start + "\n" + end, lw=3,markevery=100)

    # plt.plot(epoch_vals, y_vals, "go", label="blockchain growth in between:\n" + start + "\n" + end, lw=3, markevery=1000)

    
    # -- REGRESSION
    
    regression = []
    regression.append(r"$f_{g}(sec)$")
    regression.append(2)
    new_x, new_y, f = polynomial_interpolation(regression[0], epoch_vals, y_vals, regression[1])
    f = from_f_to_math(f)
    #plt.plot(new_x, new_y, "b-", label=regression[0], lw=3)

    
    # second regression
    
    # create matrix versions of these arrays
    x_btc_info = np.asarray(x_btc_info)
    y_btc_info = np.asarray(y_btc_info)

    x_plot = x_btc_info
    y_plot = y_btc_info
    x = x_plot
    y = y_plot

    X = x[:, np.newaxis]
    X_plot = x_plot[:, np.newaxis]

    colors = ['teal', 'yellowgreen', 'gold']
    lw = 2
    degree = 4

    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model.fit(X, y)
    y_plot = model.predict(X_plot)
    #plt.plot(x_plot, Y_plot, color="gold", linewidth=lw,
    #             label="degree %d" % degree)
    
    # -------------

    # ==== ACCURACY TEST =====
    # function \frac{9}{10^8}x^2 + \frac{5}{10^{3}}x
    h_start = 1940.0
    size_start = 10.139
    h_start = 0
    size_start = 0
    x_vals = get_list_from_file("creation_time")
    x_vals[:] = [float(x) for x in x_vals]
    x_vals[:] = [x / (60 * 60) for x in x_vals]  # in hours
    x_vals = create_growing_time_list(x_vals, h_start)
    predicted = []

    for el in x_vals:
        val_predicted = f_g(el)
        predicted.append(val_predicted)
    real = y_vals

    real[:] = [float(x) for x in real]
    real[:] = [x + size_start for x in real]
    predicted[:] = [float(x) for x in predicted]


    print mean_absolute_error(real, predicted)

    print real[-1]
    print predicted[-1]

    # === plot accuracy
    plt.plot(x_vals, real, "r-", label="real growth", lw=4)
    plt.plot(x_vals, predicted, "g-", label="predicted $f_g(x)$", lw=4)
    # =================

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 18}

    matplotlib.rc('font', **font)

    x_labels = axes.get_xticks().tolist()
    x_labels[:] = [int(x) for x in x_labels]
    start_epoch = epoch_vals[0]
    x_labels[:] = [(x*60*60) + int(start_epoch) for x in x_labels]
    x_labels[:] = [epoch_datetime(x) for x in x_labels]
    i = 0
    current = time.localtime()
    for date in x_labels:
        date = time.strptime(date, "%d-%m-%Y %H:%M:%S")
        yy = date.tm_year
        mm = calendar.month_name[date.tm_mon][:3]
        x_labels[i] = mm + str(yy)[-2:]
        i += 1

    axes.set_xticklabels(x_labels, rotation=45)

    plt.ylabel("size (GB)")
    #plt.xlabel("time (h)")
    plt.legend(loc="best")
    plt.savefig('plot/growth', bbox_inches='tight', dpi=500)
    print("plot growth.png created")
    """
# ============= GROWTH BLOCKCHAIN =================

# ============= TRANSACTION'S LATENCY ===============
    plt.figure(0)
    axes = plt.gca()
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.xaxis.set_ticks_position('bottom')
    axes.yaxis.set_ticks_position('left')



    """
    x_vals = fee_list
    y_vals = approval_time_list
    x_vals[:] = [float(x) for x in x_vals]
    x_vals[:] = [x / 100000000 for x in x_vals]  # in BTC
    print len(y_vals)
    y_vals[:] = [float(x) for x in y_vals]
    y_vals[:] = [x/(3600) for x in y_vals]   # in hours
    """

    #plt.plot(y_vals, "ro", label="% of fee paid")
    df = pd.DataFrame.from_csv(txs_dataset, sep='\t')

    block_epoch = df['B_ep'].values
    start = epoch_datetime(block_epoch[-1])
    end = epoch_datetime((block_epoch[0]))

    block_epoch[:] = [int(x) for x in block_epoch]

    print ("transactions evaluated in between:\n" + str(start) + "\n" + str(end))
    print ("\n" + "number of transactions: " + str(len(block_epoch)))

    latency = df['t_l'].values
    latency[:] = [float(x) for x in latency]

    percentage = calculate_percentage_txs_fee()

    #plt.plot(block_epoch, percentage, "r-", label="% fee paid on the net total amount")
    # --- put data (epoch) in plot form epoch to datetime

    #x_labels = axes.get_xticks().tolist()
    #x_labels = date_to_plot(x_labels)
    #axes.set_xticklabels(x_labels, rotation=45)

    # -------------


    #t_q = df['t_q'].values
    #t_f = df['t_f'].values

    #together = zip(t_q, t_f)
    #sorted_together = sorted(together)

    #t_q = [x[0] for x in sorted_together]
    #t_f = [x[1] for x in sorted_together]

    #plt.plot(t_q, "-r", label="transaction size")
    #plt.plot(t_f, "-g", label="transaction fee")

    #sns.boxplot(df.t_f, groupby=df.B_ep)
    #sns.pointplot(x="B_ep", y="t_f", data=df)
    #sns.heatmap(df.corr(), annot=True, fmt=".2f")
    #sns.jointplot(np.asarray(x_vals), np.asarray(y_vals), kind="reg", stat_func=None, color="g", xlim=(0.0, 200000.0), ylim=(0.0, 100.0))
    #plt.plot(x_vals, y_vals, "ro", label="transaction visibility")


    # ----- REGRESSION ------
    """
    regression = []
    regression.append(r"$f_{t_f}(t_f)$")
    regression.append(1)
    predicted = []
    
    mean_absolute_error_list = []
    real = y_vals
    i = 1
    while i <= 40:
        new_x, new_y, f = polynomial_interpolation(regression[0], x_vals, y_vals, i)
        predicted = []
        for el in x_vals:
            x = float(el)
            pred_y = f(x)

            predicted.append(pred_y)
        # new_y are the predicted
        # predicted = new_y

        # calculate the mean absolute error
        mean_absolute_error_list.append(mean_absolute_error(real, predicted))

        i += 1

    max_value = min(mean_absolute_error_list)
    max_index = mean_absolute_error_list.index(max_value)
    print mean_absolute_error_list
    print "best mean: " + str(max_value) + "\n"
    print "polynom: " + str(max_index)
    
    new_x, new_y, f = polynomial_interpolation(regression[0], x_vals, y_vals, regression[1])
    for el in x_vals:
        x = float(el)
        pred_y = f(x)

        predicted.append(pred_y)
    """
    #print "MAE: " + str(mean_absolute_error(y_vals, predicted)) + "\n"
    #print "\nmin tx app time: " + str(min(y_vals)) + "\n"
    #print "max tx app time: " + str(max(y_vals))

    """epoch_list[:] = [int(x) for x in epoch_list]
    epoch_list.sort()

    print epoch_datetime(epoch_list[0])
    print epoch_datetime(epoch_list[-1])"""

    #plt.plot(new_x, new_y, "g-", label=regression[0], lw=4)
    #sns.jointplot(np.asarray(x_vals), np.asarray(y_vals), kind="kde", xlim=(0.0, 0.01), ylim=(0.0, 8.0))
    #sns.jointplot(np.asarray(x_vals), np.asarray(y_vals), kind="reg", stat_func = spearmanr, color = "g", xlim=(0.0, 0.05), ylim=(0.0, 9.5), size=8, scatter=False)

    #sns.regplot(np.asarray(x_vals), np.asarray(y_vals), scatter=False)

    #axes.set_ylim([0, 5])
    #axes.set_xlim([0, 100000])
    plt.legend(loc="best")
    #plt.xlabel("txs")
    #plt.ylabel("$t_{f}$ %")
    plt.savefig('plot/fee_percentile', bbox_inches='tight', dpi=500)
    print("plot fee_percentile.png created")
    #plt.xlabel("$t_f$ (BTC)")
    #plt.ylabel("${t_l}$ (h)")
    #plt.savefig('plot/fee-approvaltime', bbox_inches='tight', dpi=500)
    #print("plot fee-approvaltime.png created")
# ============= TRANSACTION'S LATENCY ===============

# ============= AVERAGE TRANSACTION TIME ===============
    """
    plt.figure(0)
    axes = plt.gca()
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.xaxis.set_ticks_position('bottom')
    axes.yaxis.set_ticks_position('left')

    avg_approval_list = get_list_from_file("avgttime")
    epoch_list = get_list_from_file("epoch")

    avg_approval_list[:] = [float(x) for x in avg_approval_list]
    avg_approval_list[:] = [x / (60*60) for x in avg_approval_list]
    plt.plot(avg_approval_list, "r-", label="average approval time\n" + epoch_datetime(float(epoch_list[-1])) + "\n" + epoch_datetime(float(epoch_list[0])))

    axes.set_ylim([0,400])

    plt.legend(loc="best")
    plt.xlabel("block")
    plt.ylabel("${t_l}$ (h)")
    plt.savefig('plot/avg-approvaltime')
    print("plot avg-approvaltime.png created")
    """
# ============= AVERAGE TRANSACTION TIME ===============


# ================ PROPAGATION RATES AND ORPHANING ===================
    """plt.figure(0)
    axes = plt.gca()
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.xaxis.set_ticks_position('bottom')
    axes.yaxis.set_ticks_position('left')

    propagation_time = []
    orphaning_list = []

    size_list = np.linspace(1, 100000000, 100000000/100)

    size_list[:] = [float(x) for x in size_list]


    for el in size_list:
        tau = propagation_time_function(el)
        propagation_time.append(tau)
        orphan = 1 - math.exp(-( tau / 600))
        orphaning_list.append(orphan)



    size_list[:] = [x / 1000000 for x in size_list]
    propagation_time[:] = [x/(60*60) for x in propagation_time]

    plt.plot(size_list, orphaning_list, "-r", label="chance of orphaning", lw=3)


    font = {'family': 'normal',
            'weight': 'bold',
            'size': 18}

    matplotlib.rc('font', **font)

    plt.legend(loc="best")
    plt.xlabel("$Q$ (MB)")

    axes.set_xlim([0, 20])

    plt.ylabel(r"$\mathbb{P}_{orphan}$")
    plt.savefig('plot/propagation_time', bbox_inches='tight', dpi=500)
    print("plot orphaning.png created")
"""
# ================ PROPAGATION RATES AND ORPHANING ===================

# ========================== THROUGHPUT =============================
    """
    plt.figure(0)
    axes = plt.gca()
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.xaxis.set_ticks_position('bottom')
    axes.yaxis.set_ticks_position('left')

    tB = []                 # transactions approved in one block
    T = []                  # block creation time
    throughput_list = []    # list of tB/T
    e = []                  # epoch list, just to keep the lists ordered

    e, T, tB = get_lists_ordered("epoch", "creation_time", "transactions")

    tB[:] = [int(x) for x in tB]

    T[:] = [int(x) for x in T]

    for x, y in zip(tB, T):
        if(y == 0):
            y = 1
        throughput_list.append(x/y)
    avg_throughput = np.mean(throughput_list)
    avg_throughput = "{0:.2f}".format(float(avg_throughput))
    print "average throughput: " + str(avg_throughput) + " t/sec"

    plt.plot(e, throughput_list, "-b", label="Throughput\navg: " + str(avg_throughput) + " $t/sec$", lw=2)

    # --- put data (epoch) in plot
    x_labels = axes.get_xticks().tolist()
    x_labels[:] = [int(x) for x in x_labels]

    x_labels[:] = [epoch_datetime(x) for x in x_labels]
    i = 0
    for date in x_labels:
        date = time.strptime(date, "%d-%m-%Y %H:%M:%S")
        yy = date.tm_year
        mm = calendar.month_name[date.tm_mon][:3]
        dd = date.tm_mday
        x_labels[i] = str(dd) + "-" + mm + "-" + str(yy)[-2:]
        i += 1
    # -------------

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 18}

    matplotlib.rc('font', **font)

    axes.set_xticklabels(x_labels, rotation=45)
    plt.legend(loc="best")
    plt.xlabel("blocks")

    axes.set_ylim([0, 1000])
    axes.set_xlim([int(e[0]), int(e[-1])])

    plt.ylabel("throughput $t/sec$")
    plt.savefig('plot/throughput', bbox_inches='tight', dpi=500)
    print("plot throughput.png created")

"""
# ========================== THROUGHPUT =============================

# ========================= BLOCK SIZE ==============================
"""
    plt.figure(0)
    axes = plt.gca()
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.xaxis.set_ticks_position('bottom')
    axes.yaxis.set_ticks_position('left')

    e, T, Q = get_lists_ordered("epoch", "creation_time", "size")

    Q[:] = [float(x) for x in Q]
    Q[:] = [x/1000000 for x in Q]

    plt.plot(e, Q, "ob", label="Block size $Q$")

    # --- put data in plot
    x_labels = axes.get_xticks().tolist()
    x_labels[:] = [int(x) for x in x_labels]

    x_labels[:] = [epoch_datetime(x) for x in x_labels]
    i = 0
    for date in x_labels:
        date = time.strptime(date, "%d-%m-%Y %H:%M:%S")
        yy = date.tm_year
        mm = calendar.month_name[date.tm_mon][:3]
        dd = date.tm_mday
        x_labels[i] = str(dd) + "-" + mm + "-" + str(yy)[-2:]
        i += 1
    # -------------



    font = {'family': 'normal',
            'weight': 'bold',
            'size': 18}

    axes.set_xticklabels(x_labels, rotation=45)
    plt.xlabel("date")
    plt.ylabel("$Mb$")

    axes.set_ylim([0, 1.3])

    plt.legend(loc="best")
    matplotlib.rc('font', **font)
    plt.savefig('plot/block_size', bbox_inches='tight', dpi=500)
    print("plot block_size.png created")
"""
# ========================= BLOCK SIZE ==============================

def f_g(x):
    x_2 = x * x
    a = 9.0/100000000
    b = 55.0/10000

    to_return = a * x_2 + b * x

    return to_return


def propagation_time_function(q):
    """
    :param q : block size
    :return : propagation time value
    """
    if (q < 20000):
        to_return = 12.6
    else:
        q = (q / 1000) - 20
        to_return = 12.6 + (0.08 * q)

    return to_return

def calculate_profit(fee, creation_time, miner):
    """
    return the profit of a single block, knowing the creation time and the fee paid from the transactions.
    We consider 2 type of miners
    1 - Hashing power of 100 MH/s and consumption of 6.8 Watt
    2 - Hashing power of 25200 MH/s and consumption of 1250 Watt
    :param fee: M money from transactions, in BTC
    :param creation_time: time to mine a block in seconds
    :return: profit in BTC
    """

    profit = 0
    reward = 12.5
    propagation_time = 10.0
    bitcoin_hash_rate = 3700000000000000.0
    #bitcoin_hash_rate = 5061202213000000.0

    if (miner == 1):  # Smallest miner
        cost_x_hash = 4.7/1000000000000    # in USD
        hashing_rate = 100000.0
    elif (miner == 2):
        hashing_rate = 25200000.0
        cost_x_hash = 3.4/1000000000000    # in USD

    elif (miner == 3):  # AntPool - Biggest miners
        cost_x_hash = 1.091/1000000000000000
        hashing_rate = 14000000000.0

    elif (miner == 4):  # second biggest miner working
        cost_x_hash = 7.778/1000000000000000
        hashing_rate = 700000000.0

    cost_x_hash = cost_x_hash / latest_price    # in BTC
    p_orphan = 1 - math.exp(-(propagation_time/creation_time))
    cost = cost_x_hash * hashing_rate * creation_time
    revenue = (reward + fee) * (hashing_rate/bitcoin_hash_rate) * (1 - p_orphan)

    profit = revenue - cost

    return profit, revenue, cost


def save_transactions(n, hash):
    """
    retrieve n blocks but only saves transactions
    :param n: 
    :return: 
    """
    if (os.path.isfile(file_tx)):
        os.remove(file_tx)

    # -------- PROGRESS BAR -----------
    index_progress_bar = 0
    prefix = 'Saving Transactions:'
    progressBar(index_progress_bar, prefix, n)
    # ---------------------------------

    current_block = get_json_request(block_hash_url + hash)

    hash = current_block['prev_block']
    current_block = get_json_request(block_hash_url + hash)

    i = 0
    for i in range(n):
        # ---------- PROGRESS BAR -----------
        index_progress_bar += 1
        progressBar(index_progress_bar, prefix, n)
        # -----------------------------------

        txs = current_block['tx']
        # write transactions in file transactions.txt
        with io.FileIO(file_tx, "a+") as file:
            file.write(str(txs))
            file.write("\n" + str(current_block['time']) + "\n")
        i += 1
        hash_prev_block = current_block['prev_block']


        prev_block = get_json_request("https://blockchain.info/block-index/" + str(hash_prev_block) + "?format=json")

        current_block = prev_block


def get_blockchain(number_of_blocks, hash):
    """
    it retreives blocks from blockchain, given an hash where to start.

    :param number_of_blocks: int, blocks to retrieve
    :param hash: str, hash of the block from where start the retrieval
    :return: none
    """
    fetch_time_list = []
    epoch_list = []
    creation_time_list = []
    fee_list = []
    hash_list = []
    size_list = []
    height_list = []
    bandwidth_list = []
    avg_transaction_list = []
    list_transactions = []
    list_miners = []
    list_received_time = []

    # -------- PROGRESS BAR -----------
    index_progress_bar = 0
    prefix = 'Saving Blockchain:'
    progressBar(index_progress_bar, prefix, number_of_blocks)
    # ---------------------------------


    # ================== RETRIEVE BLOCKS ==================
    # retrieve blocks using json data from blockchain.info API


    current_block = get_json_request(block_hash_url + hash)

    hash = current_block['prev_block']
    start_time = datetime.datetime.now()
    current_block = get_json_request(block_hash_url + hash)
    end_time = datetime.datetime.now()

    i = 0
    for i in range(number_of_blocks):
        # ---------- PROGRESS BAR -----------
        index_progress_bar += 1
        progressBar(index_progress_bar, prefix, number_of_blocks)
        # -----------------------------------

        time_to_fetch = end_time - start_time
        time_in_seconds = get_time_in_seconds(time_to_fetch)
        fetch_time_list.append(time_in_seconds)

        # start_list, end_list = create_interval_lists()

        miner = "None"

        epoch = current_block['time']
        epoch_list.append(epoch)

        hash = current_block['hash']
        hash_list.append(hash)

        fee = current_block['fee']
        fee_list.append(fee)

        size = current_block['size']
        size_list.append(size)

        height = current_block['height']
        height_list.append(height)

        avg_tr = get_avg_transaction_time(current_block)
        avg_transaction_list.append(avg_tr)

        block_size = float(size) / 1000000  # -------> calculate read Bandwidth with MB/s
        bandwidth = block_size / time_in_seconds
        bandwidth_list.append(bandwidth)

        transactions = len(current_block['tx'])
        list_transactions.append(transactions)

        hash_prev_block = current_block['prev_block']

        start_time = datetime.datetime.now()  # ------------------------------------------------------------------------
        prev_block = get_json_request("https://blockchain.info/block-index/" + str(hash_prev_block) + "?format=json")
        end_time = datetime.datetime.now()  # --------------------------------------------------------------------------

        prev_epoch_time = prev_block['time']
        current_creation_time = int(current_block['time']) - int(prev_epoch_time)
        creation_time_list.append(current_creation_time)

        # todo: relayed by and received by give 'KeyError'
        # miner = current_block['relayed_by']
        list_miners.append(miner)

        # received_time = current_block['received_time']
        received_time = epoch
        list_received_time.append(received_time)

        current_block = prev_block

    to_write_list = [hash_list, epoch_list, creation_time_list, size_list, fee_list, height_list, bandwidth_list, list_transactions, avg_transaction_list, list_miners, list_received_time]


    # writing all the data retrieved in the file

    write_blockchain(to_write_list)

    # check blockchain status
    print blockchain_info()

def write_blockchain(to_write_list):
    """
    @param to_write_list        - Required: list to_write_list: it contains all the lists that need to be written:
        [0] hash: hash list
        [1] epoch: epoch list
        [2] creation_time: creation time list
        [3] size: size list
        [4] fee: fee list
        [5] height: height list
        [6] bandwidth: bandwidth list
        [7] transactions: number of transactions in every block list
        [8] avg_tr_list: list with the average time that a transaction need to be visible in the blockchain in a certain block
        [9] list_miner: list with all the miners for each block
        [10] list_received_time: list with all the received time for each block
    """
    n = len(to_write_list[0])
    # ---------- PROGRESS BAR -----------
    index_progress_bar = 0
    prefix = 'Writing .txt file:'
    progressBar(index_progress_bar, prefix, n)
    # -----------------------------------

    with io.FileIO(file_name, "a+") as file:
        for i in range(n):
            # --- WRITE IN FILE ---
            write_file(to_write_list, file, i)
            # ---------------------

            # -------- PROGRESS BAR -----------
            index_progress_bar += 1
            progressBar(index_progress_bar, prefix, n)
            # ---------------------------------


def write_file(list_to_write, file, index):
    """
    write the list_to_write in the file
    :param list_to_write: list containing all the other list that need to be written in the blockchain file
    :param file: open file used in write_blockchain() method
    :param index: index of which element needs to be written
    :return: none
    """
    file.write("hash: " + str(list_to_write[0][index]) + "\nepoch: " + str(list_to_write[1][index]) + "\ncreation_time: " + str(
        list_to_write[2][index]) + "\nsize: " + str(list_to_write[3][index]) + "\nfee: " + str(
        list_to_write[4][index]) + "\nheight: " + str(list_to_write[5][index]) + "\nbandwidth: " + str(
        list_to_write[6][index]) + "\ntransactions: " + str(list_to_write[7][index]) + "\navgttime: " + str(
        list_to_write[8][index]) + "\nmined_by: " + str(list_to_write[9][index]) + "\nreceived_time: " + str(list_to_write[10][index])+"\n\n")



def get_earliest_hash():
    """
    if exists, get the earliest hash saved in the blockchain local file
    :return: the earliest hash in the local blockchain file - empty string if file doesn't exist
    """
    earliest_hash = ""
    if (os.path.isfile(file_name)):
        hash_list = get_list_from_file("hash")
        if (hash_list != False):
            earliest_hash = hash_list[-1]
        else:
            earliest_hash = False
    return earliest_hash

def get_list_from_file(attribute):
    """
        return a list of "attribute" values for all the blocks in blockchain_new.txt

        :param str attribute: it could be every attribute of a block such as "size", "epoch", "hash" ...
        :return: a list containing the attribute for all the blocks

     """

    list_to_return = []

    if (os.path.isfile(file_name)):
        # open the file and read in it
        with open(file_name, "r") as blockchain_file:
            for line in blockchain_file:
                # regular expression that puts in a list the line just read: ['hash', '<value>']
                list = line.split(": ")
                list[-1] = list[-1].strip()
                # list = re.findall(r"[\w']+", line) # old regex
                if ((list) and (list[0] == attribute)):
                    list_to_return.append(list[1])
                    # print list[0] + " " + list[1]
        return list_to_return
    else:
        return False

def get_json_request(url):
    """
    Read the url and load it with json.
    :param url: str, url where to get the json data
    :return: str, the data requested in json format
    """
    json_req = urllib2.urlopen(url).read()
    request = json.loads(json_req)

    return request

def check_blockchain():
    """
    check if the element in the local blockchain have plausible datz, if not, local blockchain is not in a good status,
    in that case is better to create a new file.
     - check whether the block size is in between 1kb and 2 MB

    :return: True or False
    """
    check = True
    if (os.path.isfile(file_name)):
        list = get_list_from_file("size")
        for i in list:
            if ((int(i) > 2000000) or (int(i) < 100)):
                check = False
    return check

def blockchain_info():
    """
    print the information regarding the local blockcahin
    :return: string containing the info from the blockchain text file
    """
    string_return = ""
    if (os.path.isfile(file_name)):
        blockchain_status = check_blockchain()
        if (blockchain_status == True):
            string_return+=(bcolors.OKGREEN + "\nOK -- " + bcolors.ENDC +
                            "Blockchain checked and in a correct status.\n\nNumber of blocks:\n"
                            + '{:4}{}'.format("", str(get_number_blocks())))
        else:
            string_return+=(bcolors.FAIL + "\nFAIL -- " + bcolors.ENDC +
                            "Blockchain contains errors. Wait the end execution. If it still contains "
                            "error might be good to delete the file with -d command.\n\nNumber of blocks:\n"
                            + '{:4}{}'.format("", str(get_number_blocks())))

        list_blockchain_time = datetime_retrieved()
        string_return+=("\n\nAnalysis in between:\n" + '{:4}{}'.format("", str(list_blockchain_time[0])) + "\n" + '{:4}{}'.format("", str(list_blockchain_time[1])))

        # build the interval_string
        interval_string = blockchain_intervals()
        string_return+=("\n\nBlocks stored:\n\n" + bcolors.OKGREEN +"|| " + bcolors.ENDC + '{:^21}'.format(" HEIGHT ")
                        + bcolors.OKGREEN + "||" + bcolors.ENDC + '{:^44}'.format(" DATE ") + bcolors.OKGREEN
                        + "||" + bcolors.ENDC +"\n" + bcolors.OKGREEN + "========================================================================\n" + bcolors.ENDC + interval_string)
    else:
        string_return = "File still doesn't exist. You need to fetch blocks first with -t command.\n" + str(__doc__)
    return string_return

def get_time_in_seconds(time_to_fetch):
    """
    from time with format %H%M%S given in input return time in seconds
    :param time: time with format %H%M%S
    :return: time in seconds
    """
    # -------- TIME CONVERSION IN SECONDS ---------
    x = time.strptime(str(time_to_fetch).split('.')[0], '%H:%M:%S')
    # get the milliseconds to add at the time in second
    millisec = str(time_to_fetch).split('.')[1]
    millisec = "0." + millisec
    # get the time in seconds
    time_to_return = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()
    time_to_return = float(time_to_return) + float(millisec)
    return time_to_return

def get_avg_transaction_time(block):
    """
    get the average time, per block, of the time that a transaction
    take to be visible in the blockchain after it has been requested.

    :param block: the block to be analized
    :return: int: return the average of the time of all the transactions in the block
    """

    block_time = float(block['time'])
    tx = block['tx']

    t_sum = 0
    for t in tx:
        approval_time = block_time - int(t['time'])
        t_sum = t_sum + approval_time

    average_per_block = t_sum / len(tx)
    return average_per_block

def get_number_blocks():
    """
    :return: number of the current blocks saved in the local blockchain - 0 if file doesn't exist
    """
    number = 0
    if (os.path.isfile(file_name)):
        hash_list = get_list_from_file("hash")
        number = len(hash_list)
    return number

def datetime_retrieved(start = None, end = None):
    """
    @params:
        start   - Optional  : personalized start of the retrieval (eg, show only a portion of the blockchain)
        end     - Optional  : personalized end of the retrieved blockchain
    :return: a list containing  [0] --> end time of the blockchain retreived
                                [1] --> start time of the blockchain retrieved
    """
    # portion of the blockchain retrieved
    return_list = []
    epoch_l = get_list_from_file("epoch")
    epoch_l.sort()
    epoch_l_length = len(epoch_l)

    if(start == None):
        start = epoch_l[epoch_l_length - 1]
    else:
        start = epoch_l[start]
    if (end == None):
        end = epoch_l[0]
    else:
        end = epoch_l[end]

    start_blockchain_time = start
    end_blockchain_time = end

    start_blockchain_time = epoch_datetime(start_blockchain_time)
    end_blockchain_time = epoch_datetime(end_blockchain_time)

    return_list.append(end_blockchain_time)
    return_list.append(start_blockchain_time)

    return return_list

def epoch_datetime(epoch):
    """
    convert epoch to datetime %Y-%m-%d %H:%M:%S
    :param epoch: time in epoch
    :return: time in datetime with %Y-%m-%d %H:%M:%S format
    """
    datetime = time.strftime('%d-%m-%Y %H:%M:%S', time.localtime(float(epoch)))
    return datetime

def blockchain_intervals():
    """
    Deinfe the structure of the blockchain retrieved by displaying the hashes and their intervals taken from the
    local blockchain
    :return: a string containing the structure of the blockchain retrieved
    """
    interval_string = ""

    height_list, epoch_list = get_lists_ordered("height", "epoch")
    height_list[:] = [int(x) for x in height_list]
    epoch_list[:] = [int(x) for x in epoch_list]

    first = height_list[0]
    current = first

    i = 0
    date_start = epoch_list[i]
    date_start = epoch_datetime(date_start)

    interval_string += bcolors.OKGREEN + "|| " + bcolors.ENDC + '{:^8}'.format(str(first)) + " -- "

    for h in height_list:
        if(current == h):
            pass
        else:
            last = current - 1

            date_end = epoch_list[i-1]
            date_end = epoch_datetime(date_end)

            interval_string += '{:^8}'.format(str(last)) + bcolors.OKGREEN + " || " + bcolors.ENDC + str(date_start) \
                               + " -- " + str(date_end) + bcolors.OKGREEN + " ||\n" + "|| " + bcolors.ENDC \
                               + '{:^8}'.format(str(h)) + " -- "
            current = h

            date_start = epoch_list[i]
            date_start = epoch_datetime(date_start)

        current += 1
        i += 1
    date_end = epoch_list[-1]
    date_end = epoch_datetime(date_end)
    interval_string += '{:^8}'.format(str(height_list[-1])) + bcolors.OKGREEN + " || " + bcolors.ENDC \
                       + str(date_start) + " -- " + str(date_end) + bcolors.OKGREEN \
                       + " ||" + bcolors.ENDC + "\n"

    # write all in info.txt file
    with io.FileIO(file_info_new, "w+") as file:
        file.write(interval_string)

    return interval_string

def get_lists_ordered(name1, name2, name3 = None):
    """
    orders list of attribute names according to the height
    :return: lists ordered according to the parameters
    """
    height_list = get_list_from_file("height")
    height_list[:] = [int(x) for x in height_list]
    list1 = get_list_from_file(name1)
    list2 = get_list_from_file(name2)

    if(name3 == None):
        together = zip(height_list, list1, list2)
        sorted_together = sorted(together)

        list1 = [x[1] for x in sorted_together]
        list2 = [x[2] for x in sorted_together]
        return list1, list2
    else:
        list3 = get_list_from_file(name3)
        together = zip(height_list, list1, list2, list3)
        sorted_together = sorted(together)

        list1 = [x[1] for x in sorted_together]
        list2 = [x[2] for x in sorted_together]
        list3 = [x[3] for x in sorted_together]
        return list1, list2, list3


def polynomial_interpolation(description, x, y, degree=2):
    """
    given two lists of data it generates two new lists containing the functions interpolated
    :param  description :   description of the function
    :param  x           :   x values of the data to interpolate
    :param  y           :   y values of the data to interpolate
    :param  degree      : degree of the function to get
    :return             : x and y values to be plotted. Interpolated values. f is the function to write in the plot.
    """
    # order lists
    together = zip(x, y)
    sorted_together = sorted(together)

    x_vals = [el[0] for el in sorted_together]
    y_vals = [el[1] for el in sorted_together]

    # calculate polynomial
    z = np.polyfit(x_vals, y_vals, degree)
    f = np.poly1d(z)

    print description + ": "
    print f
    print "\n"

    x_new = np.linspace(x_vals[0], x_vals[-1], len(x_vals))
    y_new = f(x_new)


    return x_new, y_new, f

def from_f_to_math(f):
    """
    :param f    :   get a string containing the function interpolated
    :return     :   return a math string ready to be plotted
    """
    f = str(f)
    new_f = ""
    powers = []
    exp_coeff = []
    # array containing  [0]: powers
    #                   [1]: function
    array_f = f.split("\n")

    coefficients = array_f[1].split("x")
    for c in coefficients:
        exp_coeff.append(c.split("e"))

    # changing coefficients
    i = 0
    for el in coefficients:
        if (len(exp_coeff[i])>1):   # there is e^smth
            new_coeff = exp_coeff[i][0] + "*10^{" + exp_coeff[i][1] +"}"
            coefficients[i] = new_coeff
        i += 1
    exponentials = array_f[0].split(" ")
    for ex in exponentials:
        if(ex != ""):
            powers.append(ex)
    powers.append('1')

    i = 0
    for el in powers:
        new_f += coefficients[i]
        new_f += 'x^'
        new_f += el
        i += 1
    new_f += coefficients[-1]
    return new_f

def create_growing_time_list(time_list, initial_time=0):
    """
    given a time list with the creation time for each block, this method creates a new list containing the growing time
    every time a block is created.
    :param list time_list: a list with the creation time of all the blocks retrieved
    :return: list containig the growing time in hours
    """
    # create growing time list
    reversed_time_list = time_list[::-1]
    time_to_append = 0
    previous_time = initial_time
    growing_time_list = []
    #growing_time_list.append(previous_time)

    for time_el in reversed_time_list:
        time_to_append = (float(time_el)) + previous_time
        growing_time_list.append(time_to_append)
        previous_time = time_to_append

    return growing_time_list


def calculate_transactions_fee(txs, epoch = None):
    """
    given a json list of transactions, it produces the input and output fee list for each transaction in txs, plus the
    size list of each transation
    :param txs: list of transactions in json format
    :param epoch: Optional if the transaction has been approved it represents the epoch of the block in which
    this transaction is
    :return: input, output, fee, size, approval time list, txs hash list
    """
    # calculate total fee for each unconfirmed transaction
    input_fee = 0
    output_fee = 0

    in_list = []
    out_list = []
    fees_list = []
    sizes_list = []

    list_hashes_checked = []
    approval_time_list = []

    i = 0
    for tx in txs:
        try:
            sizes_list.append(tx['size'])
            # print "HASH: " + tx['hash']

            # consider a transaction only one time
            if (tx['hash'] in list_hashes_checked):
                pass
            else:
                list_hashes_checked.append(tx['hash'])
                # ===================================== GET THE TOTAL INPUT FEE ==============
                for input in tx['inputs']:
                    prev_out = input[u'prev_out']
                    input_fee += int(prev_out[u'value'])

                    # print "INPUT: " + str(prev_out[u'value'])
                in_list.append(input_fee)
                # ============================================================================

                # ===================================== GET THE TOTAL OUTPUT FEE ==============
                for output in tx['out']:
                    # print "OUTPUT: " + str(output[u'value'])

                    output_fee += int(output[u'value'])
                out_list.append(output_fee)
                # ============================================================================

                fees_list.append(float(input_fee) - float(output_fee))
                # print "FEE: " + str(float(input_fee) - float(output_fee))
                # print "APPROVAL TIME: " + str(approval_time) + "\n"
            input_fee = 0
            output_fee = 0


            # if the transactions are already approved -- calculate the approval time
            if(epoch != None):
                epoch_tx = tx['time']

                approval_time = float(epoch) - float(epoch_tx)
                approval_time_list.append(approval_time)

        except KeyError as e:
            print e
            pass
    return in_list, out_list, fees_list, sizes_list, approval_time_list, list_hashes_checked

def create_growing_size_list(size_list, initial_size=0):
    """
    given a list containig all the sizes for the blocks retrieved, create a list with the growth of the blockchain
    :param list size_list: list containing the sizes
    :return: the growth of the blockchain considering the blocks analyzed
    """
    reversed_size_list = size_list[::-1]
    growing_size_list = []
    value_to_append = 0
    size_back = initial_size
    #growing_size_list.append(value_to_append)
    # create size growing list
    for size_el in reversed_size_list:
        value_to_append = size_el + size_back
        growing_size_list.append(value_to_append)
        size_back = value_to_append

    return growing_size_list

"""
Progress bar -- from @Vladimir Ignatyev

"""
def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    """
    call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '█' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def progressBar(index_progress_bar, prefix, max_val):
    """
    :param index_progress_bar   :   number which tells where the progress are in the bar
    :param prefix               :   prefix of the progress bar
    :param max_val              :   value to reach to complete the progress bar
    """
    if(index_progress_bar == 0):
        printProgress(index_progress_bar, max_val, prefix=prefix, suffix='Complete',
                  barLength=50)
    else:
        sleep(0.01)
        printProgress(index_progress_bar, max_val, prefix=prefix, suffix='Complete',
                      barLength=50)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


if __name__ == "__main__":
    main(sys.argv[1:])
