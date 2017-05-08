# -*- coding: utf-8 -*-
"""ete011 @Enrico Tedeschi - UiT v2
Observing the Bitcoin blockchain in real time. The system will retreive portion of the Bitcoin blockchain,
do data analysis, generating models and plotting the results.

v2: the order of block retrieval doesn't matter, is possible to retrieve blocks starting from a certain hash

Usage: observ.py -t number
    -h | --help         : usage
    -i                  : gives info of the blockchain retrieved
    -t number           : checks the unconfirmed transactions and plot the mempool demand and space supply curves. Input number for how many new transactions you want to consider, [400-2000]
    -T                  : generates the plot regarding the approval time according to the fee paid for each transaction
    -p                  : plots data in .txt files
    -r                  : revores the .txt files previously backed up
    -c number           : retrieves blocks to compare the blockchain in different epoch. number is the number of blocks to be retrieved
    -C number           : same as -c but it will save also all the transactions for each block in transactions.txt file. IMPORTANT: use it only one time for the retrieval, otherwise transactions won't be ordered. You can print the results with -T command
    -d                  : delete info.txt, blockchain.txt and unconfirmed_tx.txt if exist
    -b                  : backup of info, blockchain and transactions file

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


# ------ GLOBAL ------
global file_blockchain
file_blockchain = "blockchain.txt"

global file_info
file_info = "info.txt"

global file_unconfirmed_tx
file_unconfirmed_tx = "unconfirmed_tx.txt"

global file_tx
file_tx = "transactions.txt"

global n_portions
n_portions = 4

global color_list
color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

global marker_list
marker_list = ['o', '-', '*', '^']

global bitcoin
bitcoin = u'\u0243'

global latest_block_url
latest_block_url = "https://blockchain.info/latestblock"

global unconfirmed_txs_url
unconfirmed_txs_url = "https://blockchain.info/unconfirmed-transactions?format=json"

global block_hash_url
block_hash_url = "https://blockchain.info/rawblock/"

# --------------------

def main(argv):
    try:
        global plot_number  #todo: move to the global part
        plot_number = 0
        args_list = sys.argv
        args_size = len(sys.argv)
        earliest_hash = get_earliest_hash()

        opts, args = getopt.getopt(argv, "Thipbrdt:c:C:")
        valid_args = False

        for opt, arg in opts:
            if(opt == "-t"):    # check unconfirmed transaction for the mempool demand and supply curve
                print ("Checking the unconfirmed transactions.")
                start = datetime.datetime.now()
                fetch_unconfirmed_transactions(int(arg))
                end = datetime.datetime.now()

                time = end - start
                time = get_time_in_seconds(time)
                plot_demand_supply_curve(time)
                # BLOCK SPACE SUPPLY CURVE
                valid_args = True
            if(opt == "-b"):    # backup of txt files
                print "Backup of .txt files."
                copyFiles(file_tx, "bu_"+file_tx)
                copyFiles(file_blockchain, "bu_"+file_blockchain)
                copyFiles(file_info, "bu_"+file_info)
                valid_args = True
            if(opt == "-T"):
                # analyse transactions in transactions.txt
                plot_tx_visibility()
                valid_args = True
            if(opt == "-d"): #delete info.txt and blockchain.txt
                valid_args = True
                delete_file(file_info)
                delete_file(file_blockchain)
                delete_file(file_unconfirmed_tx)
                delete_file(file_tx)
            if(opt == "-i"):    # blockchain info
                # test_accuracy()
                print blockchain_info()
                valid_args = True
            if(opt == "-h"):    # usage
                print (__doc__)
                valid_args = True
            if(opt == "-p"):    # plot only
                plot_sequence()
                valid_args = True
            if (opt == "-r"):  # recover txt files
                print "Recovering .txt files."
                copyFiles("bu_" + file_tx, file_tx)
                copyFiles("bu_" + file_blockchain, file_blockchain)
                copyFiles("bu_" + file_info, file_info)
                valid_args = True
            if(opt == "-c"):    # compare with older blocks - retrieve blockchain in previous time
                blocks = int(arg)
                define_intervals(blocks)
                valid_args = True
            if (opt == "-C"):
                blocks = int(arg)
                define_intervals(blocks, True)
                valid_args = True

        if(valid_args == False):
            print (__doc__)
    except getopt.GetoptError:
        print (__doc__)
        sys.exit(2)

def plot_multiple_lists(description, marker, epoch_list, list2, list3 = None, alternative_intervals = None, regression = None):
    """
    receive lists to be plotted, list2 and list3. They are given as a list ready to be plotted already.
    :param description              - Required  :   description of the graph to plot
    :param marker                   - Required  :   marker to plot, could be '-', 'o', '*' ecc...
    :param epoch_list               - Required  :   list with epoch to put on label
    :param list2                    - Required  :   list with x values to plot
    :param list3                    - Optional  :   second list with the y values to plot
    :param alternative_intervals    - Optional  :   if there are other intervals, like in the transactions ones
    :param regression               - Optional  :   array containing:   0: description
                                                                        1: degree
    """
    #todo: RE ORGANIZE THIS METHOD WITH THE INTERPOLATION AS WELL
    if (list3 != None and list3 != []):
        is_x = True
    else:
        is_x = False

    # retrieve the intervals to print
    start_interval, end_interval = get_indexes(epoch_list)
    if (alternative_intervals == None):
        start_interval_tr, end_interval_tr = get_indexes()
    else:
        start_interval_tr = alternative_intervals[0]
        end_interval_tr = alternative_intervals[1]

    # list to plot the n portions
    to_plot_1 = []
    to_plot_2 = []
    to_plot_3 = []

    i = 0
    while (i < n_portions):
        to_plot_1.append(epoch_list[start_interval[i]:end_interval[i]])
        to_plot_2.append(list2[start_interval_tr[i]:end_interval_tr[i]])
        if(is_x):
            to_plot_3.append(list3[start_interval_tr[i]:end_interval_tr[i]])
            # order before printing - order according to x_vals (list2)
            together_sorted = sorted(zip(to_plot_2[i], to_plot_3[i]))

            to_plot_2[i][:] = [xv[0] for xv in together_sorted]
            to_plot_3[i][:] = [yv[1] for yv in together_sorted]

            plt.plot(list2[start_interval_tr[i]:end_interval_tr[i]], list3[start_interval_tr[i]:end_interval_tr[i]], color_list[i] + marker,
                     label=(str(epoch_datetime(to_plot_1[i][0])) + "\n" + str(epoch_datetime(to_plot_1[i][-1]))))
            if(regression != None):
                new_x, new_y, f = polynomial_interpolation(regression[0], list2[start_interval_tr[i]:end_interval_tr[i]], list3[start_interval_tr[i]:end_interval_tr[i]], regression[1])
                # todo: create a function that manipulates f
                f = from_f_to_math(f)
                plt.plot(new_x, new_y, color_list[i]+marker_list[1], label="$" + f + "$", lw=4)
        else:
            plt.plot(list2[start_interval_tr[i]:end_interval_tr[i]], color_list[i] + marker,
                 label=(str(epoch_datetime(to_plot_1[i][0])) + "\n" + str(epoch_datetime(to_plot_1[i][-1]))))
        i += 1
    plt.legend(loc="best", prop={'size': 8})


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




# ACCURACY WITH NORMAL DISTRIBUTION
def test_accuracy():
    """
    Verify the accuracy of the function fBg(x) by comparing fees having a certain creation time
    """

    fee_list, creation_time_list, epoch_list = get_lists_ordered("fee", "creation_time", "epoch")

    fee_list[:] = [float(x) for x in fee_list]
    fee_list[:] = [x / 100000000 for x in fee_list]  # in BTC

    creation_time_list[:] = [float(x) for x in creation_time_list]
    creation_time_list[:] = [x / 60 for x in creation_time_list]  # in minutes

    diff_list = []
    expected_list = []

    i = 0
    for fee in fee_list:
        expected = fBg(creation_time_list[i])
        expected_list.append(expected)
        diff = fee - expected
        diff_list.append(diff)
        i += 1
    # diff_list.sort()


    # ============================== PRECISION ==============================
    # precision = max_diff - min_diff * 100% / mean(max_diff, min_diff)
    max_diff = max(diff_list)
    min_diff = min(diff_list)
    precision = ((max_diff - min_diff) * 100) / ((max_diff + min_diff) / 2)
    print "Precision: " + str(precision) + " %"

    # level of precision for each element
    precision_list = []

    i = 0
    for exp in expected_list:
        precision = ((fee_list[i] - exp) * 100) / ((fee_list[i] + exp) / 2)
        precision_list.append(precision)
        i += 1

    print np.mean(precision_list)

    # =======================================================================



    # ============================== ACCURACY ===============================
    # experimental - true * 100% / true

    accuracy_list = []
    i = 0
    print expected_list[0]
    print fee_list[0]
    for fee in fee_list:
        accuracy = (fee - expected_list[i]) * 100 / expected_list[i]
        i += 1
        accuracy_list.append(accuracy)

    print np.mean(accuracy_list)

    # =======================================================================


    # ================== NORMAL DISTRIBUTION ========================
    axes = plt.gca()

    # mu = 0
    # variance = 0.031772
    mu = np.mean(diff_list)
    variance = np.var(diff_list)
    sigma = math.sqrt(variance)

    y_l = norm.pdf(diff_list, mu, sigma)
    """ myList = ','.join(map(str, diff_list))
    print myList
    plt.plot(diff_list)
    plt.savefig('plot/normal_curve')"""
    # y_l[:] = [y * 100 for y in y_l]

    together_sorted = sorted(zip(epoch_list, diff_list, y_l))

    diff_list[:] = [xv[1] for xv in together_sorted]
    y_l[:] = [yv[2] for yv in together_sorted]
    plot_multiple_lists("accuracy", marker_list[0], epoch_list, diff_list, y_l)

    plt.legend(loc="best")

    axes.set_xlim([min(diff_list), 2])

    plt.xlabel("BTC")
    plt.ylabel("Percentage %")
    plt.savefig('plot/accuracy')

    # ===============================================================

def plot_tx_visibility():
    """
    Read the transactions.txt file and plot the relation between the fee paid and the transaction visibility
    :return:
    """

    epoch_list = []
    fee_list = []
    size_list = []
    approval_time_list = []


    epoch_list_from_info = get_list_from_file("epoch")
    epoch_list_from_info[:] = [float(x) for x in epoch_list_from_info]
    epoch_list_from_info.sort()

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
        progressBar(index_progress_bar, 'Reading ' + file_tx + ':', len(list_txs))
        # ---------------------------------

        # delete the epoch from the list just retrieved
        i = 0
        for el in list_txs:
            epoch_list.append(list_txs[i+1])
            list_txs.remove(list_txs[i+1])
            i += 1
            # ---------- PROGRESS BAR -----------
            index_progress_bar += 1
            progressBar(index_progress_bar, prefix, (len(list_txs)) + (len(epoch_list)))
            # -----------------------------------
        list_txs[:] = [ast.literal_eval(t) for t in list_txs]

        # create indexing to plot the transactions visibility with the temporal graph -- divided into n portions
        total_blocks = len(epoch_list)
        one_portion = total_blocks / n_portions
        indexes_start_list = []
        indexes_end_list = []
        number_of_transactions_each_epoch = []
        sum = 0

        index = 1
        for i in range(len(epoch_list)):
            # ---------- PROGRESS BAR -----------
            index_progress_bar += 1
            progressBar(index_progress_bar, prefix, (len(list_txs)) + (len(epoch_list)))
            # -----------------------------------

            list_txs[i].pop(0)      # remove the first transaction of each block since it is only the reward
            _, _, temp_fee_list, temp_size_list, temp_approval_time_list = calculate_transactions_fee(list_txs[i], int(epoch_list[i]))
            # print len(temp_approval_time_list)    # how many transaction per block
            sum += len(temp_approval_time_list)
            fee_list.extend(temp_fee_list)
            size_list.extend(temp_size_list)
            approval_time_list.extend(temp_approval_time_list)
            if (index % one_portion == 0):  # changed epoch
                number_of_transactions_each_epoch.append(sum)
                sum = 0
            index += 1
        print number_of_transactions_each_epoch   # list containing how many transactions per each portion

        # define transactions intevals
        for i in range(n_portions):
            if (i == 0):
                indexes_start_list.append(0)
                indexes_end_list.append(indexes_start_list[i] + number_of_transactions_each_epoch[i])
            else:
                indexes_start_list.append(indexes_end_list[i-1] + 1)
                indexes_end_list.append(indexes_start_list[i] - 1 + number_of_transactions_each_epoch[i])

        indexes = []
        print indexes_start_list    # list containing the start indexes to plot the different portion in the graph
        print indexes_end_list      # list containing the end indexes to plot the different portion in the graph
        indexes.append(indexes_start_list)
        indexes.append(indexes_end_list)

        # plot data
        c = CurrencyRates()
        b = BtcConverter()

        one_usd = b.get_latest_price('USD')
        axes = plt.gca()

        x_vals = fee_list
        x_vals[:] = [float(x) for x in x_vals]
        x_vals[:] = [x / 100000000 for x in x_vals]  # in BTC
        # y_vals[:] = [x * one_usd for x in y_vals]  # in USD

        y_vals = approval_time_list
        print len(y_vals)
        y_vals[:] = [float(x) for x in y_vals]
        y_vals[:] = [x / 60 for x in y_vals]  # in minutes

        # together = zip(x_vals, y_vals)
        # sorted_together = sorted(together)

        # x_vals = [x[0] for x in sorted_together]
        # y_vals = [x[1] for x in sorted_together]

        epoch_list[:] = [int(x) for x in epoch_list]
        epoch_list.sort()
        plot_multiple_lists("transaction visibility", marker_list[0], epoch_list, x_vals, y_vals, alternative_intervals=indexes)

        # LOG SCALE
        # plt.yscale('log', nonposy='clip')
        # plt.xscale('log', nonposy='clip')

        # axes.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
        # axes.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        axes.set_ylim([0, 100])
        axes.set_xlim([0, 0.02])
        plt.xlabel("$t_f$ (BTC)")
        plt.ylabel("$f_{t_f}(x)$ available bandwidth (min)")
        plt.legend(loc="best")
        plt.savefig('plot/fee-approvaltime')
        print("plot fee-approvaltime.png created")

    else:
        print "file " + file_tx + " still doesn't exist. Fetch some blocks with -c command first!"

def propagation_time_function(q):
    """
    :param q : block size
    :return : propagation time value
    """
    to_return = math.exp(q*1.5)
    return to_return


def fetch_unconfirmed_transactions(max_tr):
    """
    it fetches the new unconfirmed transactions according to a limit defined from max_tr
    :param max_tr : number of iterations to retrieve the unconfirmed transactions
    Rtrieve the unconfirmed transactions to analyze them, and write a file with all the unconfirmed transactions.
    return: the unconfirmed transactions
    """

    unconfirmed_tr = []

    # -------- PROGRESS BAR -----------
    index_progress_bar = 0
    prefix = 'Fetching unconfirmed Txs:'
    progressBar(index_progress_bar, prefix, max_tr)
    # ---------------------------------

    # how many unconfirmed transactions you want to retreive, 10 per time
    i = 0
    while i < max_tr:
        try:
            json_data = get_json_request(unconfirmed_txs_url)
            unconfirmed_tr += json_data['txs']
            time.sleep(1.5)
            i += 1
            # ---------- PROGRESS BAR -----------
            index_progress_bar += 1
            progressBar(index_progress_bar, prefix, max_tr)
            # -----------------------------------
        except Exception as e:
            pass

    # write file with all the unconfirmed transactions
    with io.FileIO(file_unconfirmed_tx, "w") as file:
        file.write(str(unconfirmed_tr))
    print "file " + file_unconfirmed_tx + " saved."


def calculate_transactions_fee(txs, epoch = None):
    """
    given a json list of transactions, it produces the input and output fee list for each transaction in txs, plus the
    size list of each transation
    :param txs: list of transactions in json format
    :param epoch: Optional if the transaction has been approved it represents the epoch of the block in which
    this transaction is
    :return: input, output, fee, size, approval time list
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
    return in_list, out_list, fees_list, sizes_list, approval_time_list

def plot_demand_supply_curve(time):
    """
    It plots the mempool demand and supply curve. It retrievs the unconfirmed txs in the unconfirmed_tx.txt file,
    then it calculates the fees density with fee/size. It orders it in a decrescent order and then
    plots the cumulate sum of the fees and the sizes following this order in order to get the mempool demand curve.
    :param time :   time, in second, taken checking the mempool
    """
    fees_list = []
    sizes_list = []
    fee_density_list = []

    if (os.path.isfile(file_unconfirmed_tx)):
        with io.FileIO(file_unconfirmed_tx, "r") as file:
            file.seek(0)
            unconfirmed_tx = file.read()

        unconfirmed_tx = ast.literal_eval(unconfirmed_tx)
        _, _, fees_list, sizes_list, _, = calculate_transactions_fee(unconfirmed_tx)

        # ================== MEMPOOL DEMAND CURVE ==================
        # create the fee density list to order the other two lists
        for f, s in zip(fees_list, sizes_list):
            fee_density_list.append(float(f/float(s)))

        # order according to the fee_density_list
        together = zip(fee_density_list, sizes_list, fees_list)
        sorted_together = sorted(together, reverse=True)

        fee_density_list = [x[0] for x in sorted_together]
        sizes_list = [x[1] for x in sorted_together]
        fees_list = [x[2] for x in sorted_together]

        fees_list[:] = [float(x)/100000000 for x in fees_list] # in BTC
        sizes_list[:] = [float(x)/1000000 for x in sizes_list] # IN Mb
        # growing lists
        fees_list = np.cumsum(fees_list)
        sizes_list = np.cumsum(sizes_list)

        # ================== BLOCK SPACE SUPPLY CURVE ==================
        cost_list = []
        reward = 12.5   # current reward per each block in BTC
        creation_time = 600 # assuming 10 minutes of creation time

        for el in sizes_list:
            tau = propagation_time_function(float(el))
            # formula from Rizun's Paper
            supply = reward * (math.exp(tau / creation_time) - 1)
            cost_list.append(supply)


        # find the point of maximization of the profit
        max_l = []
        x_list_max = []
        i = 0
        max_v = 0
        for fee in fees_list:
            val = fee - cost_list[i]
            max_l.append(val)
            if (max_v < val):
                max_v = val
                maximized_space = sizes_list[i]
            i += 1

        for i in range(len(max_l)):
            x_list_max.append(maximized_space)

        int(time)
        time_min = int(time / 60)
        # ================== PLOTTING ==================
        axes = plt.gca()
        axes.set_ylim([0, max(fees_list) + 1])
        axes.set_xlim([0, max(sizes_list)])
        plt.figure(1)
        plt.rc('lines', linewidth=3)
        plt.plot(sizes_list, fees_list, color_list[1] + marker_list[1],
                 label=("$M_{demand}(b)$"), )
        plt.plot(sizes_list, cost_list, color_list[2] + marker_list[1],
                 label=("$M_{supply}(Q)$"), )
        plt.plot(x_list_max, list(range(len(fees_list))), color_list[3] + marker_list[1],
                 label=("$Q^*$ for " + str(time_min) + " minutes"))
        plt.legend(loc="best", prop={'size': 8})
        plt.ylabel(r"Fees $M(B)$")
        plt.xlabel("Block space $Q$ (Mb)")
        plt.savefig('plot/demandsupplycurve', transparent=True)
        print "plot demandsupplycurve.pdf created"
        # =======================================

    else:
        print "File " + file_unconfirmed_tx + " does not exist!"


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

def get_json_request(url):
    """
    Read the url and load it with json.
    :param url: str, url where to get the json data
    :return: str, the data requested in json format
    """
    json_req = urllib2.urlopen(url).read()
    request = json.loads(json_req)

    return request


def define_intervals(number_of_blocks, save_txs = False):
    """
    Retrieves blocks with a certain interval according to how many blocks are present in the blockchain.
    These intervals are called portions.
    We set a number of portions = 4
    :param number_of_blocks
    :param save_txs: if True save also the transactions, False by default, saving transactions is expensive in terms of memory
    """
    error = False
    # define p = number of blocks per portion
    # n = number of blocks in the blockchain

    last_block = get_json_request(latest_block_url)
    n = int(last_block['height'])
    p = n / n_portions

    start_list, end_list = create_interval_lists()
    i = 0
    # case if the files doesn't exist
    if (start_list == []):
        # starting from 0

        # get the heights and hashes where to start:
        while (i < n_portions):
            if(i == 0):
                # the retrieval starts with the latest block, and then the previous are retrieved as well
                height_to_start = number_of_blocks
            else:
                height_to_start = (i*p) + number_of_blocks
            b_array = get_json_request("https://blockchain.info/block-height/" + str(height_to_start) + "?format=json")
            blocks = b_array['blocks']
            b = blocks[0]
            hash = b['hash']
            epoch = b['time']
            time = epoch_datetime(int(epoch))
            i += 1
            print "Retrieving " + str(number_of_blocks) + " starting from " + time
            get_blockchain(number_of_blocks, error, hash, save_txs)

    # case if the files exists already
    else:
        # in start_list and end_list are stored all the heights representing the intervals retrieved
        # todo: control that the number of blocks is not higher than the portion p
        if (int(end_list[0]) + number_of_blocks >= p):
            print (bcolors.WARNING + "WARNING: " + bcolors.ENDC + "Blockchain already up to date!")
        else:
            while (i < n_portions):
                height_to_start = end_list[i] + number_of_blocks
                b_array = get_json_request(
                    "https://blockchain.info/block-height/" + str(height_to_start) + "?format=json")
                blocks = b_array['blocks']
                b = blocks[0]
                hash = b['hash']
                epoch = b['time']
                time = epoch_datetime(int(epoch))
                i += 1
                print "Retrieving " + str(number_of_blocks) + " blocks starting from " + time
                get_blockchain(number_of_blocks, error, hash)

def plot_sequence():
    """

    """

    plot_data("time_per_block", 0)
    plot_data("byte_per_block", 1)
    plot_data("growth_blockchain", 2)
    plot_data("fee_bandwidth", 3)
    plot_data("bandwidth", 4)
    """plot_data("efficiency", 5)
    plot_data("fee_transactions", 7)
    plot_data("transaction_visibility", 6)
    plot_data("tthroughput", 8)"""

# @profile
def get_blockchain(number_of_blocks, error, hash, save_txs = False):
    # todo: remove the parameter 'error'
    """
    it retreives blocks from blockchain, given an hash where to start.

    :param number_of_blocks: int, blocks to retrieve
    :param error: boolean, if True data are retrieved in Json if False through the client API
    :param hash: str, hash of the block from where to start the retrieval
    :param save_txs: if True save also the transactions, False by default, saving transactions is expensive in terms of memory
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

    start_time = datetime.datetime.now()
    current_block = get_json_request(block_hash_url + hash)
    end_time = datetime.datetime.now()


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

        avg_tr = get_avg_transaction_time(current_block, True)
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

        # add_mining_nodes(current_block)

        if(save_txs == True):
            txs = current_block['tx']
        # write transactions in file transactions.txt
            with io.FileIO(file_tx, "a+") as file:
                file.write(str(txs))
                file.write("\n" + str(current_block['time']) + "\n")


        current_block = prev_block



    to_write_list = [hash_list, epoch_list, creation_time_list, size_list, fee_list, height_list, bandwidth_list, list_transactions, avg_transaction_list, list_miners, list_received_time]


    # writing all the data retrieved in the file

    write_blockchain(to_write_list)

    # check blockchain status
    print blockchain_info()

# @profile
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

    with io.FileIO(file_blockchain, "a+") as file:
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


def get_creation_time(currblock, prevblock, isJson, error_local, currtime):
    """
    Block that get the creation time, and it considers the possible negative
    creation time due to a time error in the blockchain.
    Apparently some blocks have the previous block with an higher timestamp, so this method consider the
    previous block the first one with a lower timestamp if the creation time turns to be negative.
    :param currblock: current block to analyze
    :param prevblock: block where to start the search of the ancestor
    :param isJson: bool: if True data must be collected through Json
    :param error_local: bool: tells if the block being analyzed is inside an error catch
    :return: return the positive creation time for the currblock
    """
    right_ancestor = prevblock
    if (isJson == False):
        while (currtime < 0):
            right_ancestor = blockexplorer.get_block(right_ancestor.previous_block)
            currtime = currblock.time - right_ancestor.time
            print "creation time turned positive: " + str(currtime)
    elif (isJson == True):
        while (currtime < 0):
            right_ancestor = get_json_request("https://blockchain.info/block-index/" + prevblock["prev_block"] + "?format=json")
            if(error_local == False):
                currtime = currblock.time - right_ancestor["time"]
            elif (error_local == True):
                currtime = currblock["time"] - right_ancestor["time"]
            print "creation time turned positive: " + str(currtime)
    return currtime


def create_growing_time_list(time_list):
    """
    given a time list with the creation time for each block, this method creates a new list containing the growing time
    every time a block is created.
    :param list time_list: a list with the creation time of all the blocks retrieved
    :return: list containig the growing time in hours
    """
    # create growing time list
    reversed_time_list = time_list[::-1]
    time_to_append = 0
    previous_time = 0
    growing_time_list = []
    growing_time_list.append(previous_time)

    for time_el in reversed_time_list:
        # time in hours
        time_to_append = (float(time_el) / (60 * 60)) + previous_time
        growing_time_list.append(time_to_append)
        previous_time = time_to_append

    return growing_time_list


def create_growing_size_list(size_list):
    """
    given a list containig all the sizes for the blocks retrieved, create a list with the growth of the blockchain
    :param list size_list: list containing the sizes
    :return: the growth of the blockchain considering the blocks analyzed
    """
    reversed_size_list = size_list[::-1]
    growing_size_list = []
    value_to_append = 0
    size_back = 0
    growing_size_list.append(value_to_append)
    # create size growing list
    for size_el in reversed_size_list:
        value_to_append = size_el + size_back
        growing_size_list.append(value_to_append)
        size_back = value_to_append

    return growing_size_list


def get_list_from_file(attribute):
    """
        return a list of "attribute" values for all the blocks in blockchain.txt

        :param str attribute: it could be every attribute of a block such as "size", "epoch", "hash" ...
        :return: a list containing the attribute for all the blocks

     """

    list_to_return = []

    if (os.path.isfile(file_blockchain)):
        # open the file and read in it
        with open(file_blockchain, "r") as blockchain_file:
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


def get_avg_transaction_time(block, json):
    """
    get the average time, per block, of the time that a transaction
    take to be visible in the blockchain after it has been requested.

    :param block: the block to be analized
    :param json: if true, the json version of the block needs to be analyzed
    :return: int: return the average of the time of all the transactions in the block
    """

    if(json == True):
        block_time = float(block['time'])
        tx = block['tx']

        t_sum = 0
        for t in tx:
            approval_time = block_time - int(t['time'])
            t_sum = t_sum + approval_time

        average_per_block = t_sum / len(tx)
    else:
        # take transactions the block
        transactions = block.transactions

        # get block time -- when it is visible in the blockchain, so when it was created
        block_time = block.time

        # list of time of each transactions in one block
        transactions_time_list = []

        # list of the time that each transaction take to be visible, so when the block is visible in the blockchain
        time_to_be_visible = []

        for t in transactions:
            transactions_time_list.append(float(t.time))

        for t_time in transactions_time_list:
            time_to_be_visible.append(float(block_time - t_time))

        average_per_block = sum(time_to_be_visible) / len(time_to_be_visible)
    return average_per_block


def add_mining_nodes(block):
    """
    given a block add in a file all the new mining nodes and nodes involved in relying a transaction
    :param block: each block has a number of transactions and these transactions are relayed by a node
    :return: None
    """

    nodes_list = []
    nodes_list_new = []

    # if file already exist then read it and make a list of nodes out of it, this list will be appended to the file later
    with io.FileIO("nodes_in_the_network.txt", "a+") as file:
        if (os.path.isfile('nodes_in_the_network.txt')):
            file.seek(0)
            # get the list from the file
            for line in file:
                line = line.split()[0]
                nodes_list.append(line)

        transactions = block.transactions
        for t in transactions:
            node = str(t.relayed_by)
            if (node in nodes_list):
                pass
                # print node + " in list"
            elif(node in nodes_list_new):
                pass
                # print node + " in list new"
            else:
                nodes_list_new.append(node)

        for n in nodes_list_new:
            file.write(n + "\n")


    # for the mining nodes file, so the file containing the nodes which relay blocks
    node_list = []

    with io.FileIO("mining_nodes.txt", "a+") as file:
        if (os.path.isfile('nodes_in_the_network.txt')):
            file.seek(0)
            # get the list from the file
            for line in file:
                line = line.split()[0]
                nodes_list.append(line)
        node = str(block.relayed_by)
        if (node in nodes_list):
            pass
        else:
            file.write(node + "\n")

"""
Plotting

defined methods:
    - plot_blockchain(list, str, str)
"""


def get_indexes(list = []):
    """
    get the start and end indexed where the list needs to be splitted
    :return : list with indexes
    """
    interval_list_start = []
    interval_list_end = []
    if(list != []):
        l = len(list)
    else:
        l = len(get_list_from_file("height"))

    portion = l / n_portions

    i = 0
    start = 0
    end = int(portion)
    while (i < n_portions):
        interval_list_start.append(start)
        interval_list_end.append(end)

        start = end
        end = start + portion
        i += 1
    return interval_list_start, interval_list_end


def plot_data(description, plot_number):
    """
    Get the lists in the file and plots the data according to the description.
    :param description  - Required  : describe the type of plot created, it might be:
        time_per_block
        byte_per_block
        bandwidth
        growth_blockchain
        transaction_visibility
        efficiency
        fee_bandwidth
    :param plot_number  - Required  : number of the plot to be plotted and saved (progressive number)
    """
    plt.figure(plot_number)
    plt.rc('lines', linewidth=1)
    axes = plt.gca()
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.xaxis.set_ticks_position('bottom')
    axes.yaxis.set_ticks_position('left')

    if(description == "time_per_block"): # shows the creation time for each block
        y_vals, x_vals = get_lists_ordered("creation_time", "epoch")
        x_vals[:] = [float(x) for x in x_vals]
        y_vals[:] = [float(y) for y in y_vals]

        y_vals[:] = [y / 60 for y in y_vals]
        plot_multiple_lists(description, marker_list[0], x_vals, y_vals)

        plt.ylabel("creation time (min)")
        plt.xlabel("epoch")
        axes.set_ylim([0, 40])

        plt.savefig('plot/' + description + '(' + str(len(x_vals)) + ')')

        print("plot " + description + ".png created")
    elif(description == "byte_per_block"): # shows the size of the blocks during time
        y_vals, x_vals = get_lists_ordered("size", "epoch")
        y_vals[:] = [float(i) for i in y_vals]
        y_vals[:] = [y / 1000000 for y in y_vals]

        plot_multiple_lists(description, marker_list[0], x_vals, y_vals)

        plt.ylabel("block size (Mb)")
        plt.xlabel("block number")
        axes.set_xlim([0, len(x_vals)/n_portions])
        max_in_list = max(y_vals)
        axes.set_ylim([0, max_in_list*1.4])

        """# label of the time
        at = AnchoredText(list_blockchain_time[0],prop=dict(size=8), frameon=True,loc=3,)
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        axes.add_artist(at)

        at = AnchoredText(list_blockchain_time[1], prop=dict(size=8), frameon=True, loc=4,)
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        axes.add_artist(at)
        # end label of the time"""

        plt.savefig('plot/' + description + '(' + str(len(x_vals)) + ')')
        print("plot " + description + ".png created")
    elif(description == "bandwidth"): # shows the read bandwidth of the blockchain -- how much time to retrieve data
        y_vals, x_vals = get_lists_ordered("bandwidth", "epoch")
        y_vals[:] = [float(y) for y in y_vals]

        plot_multiple_lists(description, marker_list[1], x_vals, y_vals)

        # plt.plot(x_vals, 'c-', label=(
            # "read bandwidth Mb/s\n" + str(list_blockchain_time[0]) + "\n" + str(list_blockchain_time[1])), lw=3)

        plt.ylabel("read bandwidth (Mb/s)")
        plt.xlabel("block number")
        axes.set_xlim([0, len(y_vals)/n_portions])
        max_in_list = max(y_vals)
        axes.set_ylim([0, max_in_list * 1.2])

        plt.savefig('plot/' + description + '(' + str(len(x_vals)) + ')')
        print("plot " + description + ".png created")
    elif(description == "growth_blockchain"):
        epoch_vals, x_vals, y_vals = get_lists_ordered("epoch", "creation_time", "size")

        x_vals[:] = [float(x) for x in x_vals]
        # x_vals[:] = [x / (60 * 60) for x in x_vals]  # in hours

        y_vals[:] = [float(x) for x in y_vals]
        y_vals[:] = [y / 1000000000 for y in y_vals]  # in GB

        plt.ylabel("size (GB)")
        plt.xlabel("time (h)")
        # plot_multiple_lists("growth blockchain", marker_list[0], epoch_vals, x_vals, y_vals, cumulate=True)
        # axes.set_xlim([0, max(x_vals)])

        # split the list in epoch and create a growing size and time list
        start, end = get_indexes()
        i = 0
        size_lists = []
        time_lists = []
        size = []
        time = []

        while (i < n_portions):
            size_lists.append(y_vals[start[i]:end[i]])
            time_lists.append(x_vals[start[i]:end[i]])

            # create growing size lists
            size_lists[i] = create_growing_size_list(size_lists[i])
            time_lists[i] = create_growing_time_list(time_lists[i])
            size.extend(size_lists[i])
            time.extend(time_lists[i])

            size.pop()  # pop last element because of the 0 at the beginning
            time.pop()
            i += 1

        x_vals = time
        y_vals = size
        regression = []
        regression.append("growth regression")
        regression.append(2)
        plot_multiple_lists("growth blockchain", marker_list[0], epoch_vals, x_vals, y_vals, regression=regression)
        # plt.yscale('log', nonposy='clip')
        # plt.xscale('log', nonposy='clip')
        # axes.yaxis.set_major_formatter(FormatStrFormatter('%.5f'))
        # axes.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.savefig('plot/' + description + '(' + str(len(x_vals)) + ')')
        print("plot " + description + ".png created")
    elif(description == "transaction_visibility"):
        x_vals = get_list_from_file("avgttime")
        x_vals[:] = [float(x) for x in x_vals]
        x_vals[:] = [x / 60 for x in x_vals]    # in minutes
        plt.plot(x_vals, 'b-', label=(
        "avg transaction visibility per block\n" + str(list_blockchain_time[0]) + "\n" + str(list_blockchain_time[1])))
        plt.legend(loc="best")
        plt.ylabel("time (min)")
        plt.xlabel("block number")
        axes.set_xlim([0, len(x_vals)])

        plt.savefig('plot/' + description + '(' + str(len(x_vals)) + ')')
        print("plot " + description + ".png created")
    elif(description == "efficiency"):
        x_vals_size = get_list_from_file("size")
        x_vals_time = get_list_from_file("creation_time")
        x_vals_tr = get_list_from_file("transactions")

        x_vals_size[:] = [float(x) for x in x_vals_size]
        x_vals_size[:] = [x / 1000 for x in x_vals_size]
        x_vals_time[:] = [float(x) for x in x_vals_time]
        x_vals_tr[:] = [float(x) for x in x_vals_tr]

        plt.plot(x_vals_time, 'b-', label="Block Creation Time (sec)", lw=3)
        plt.plot(x_vals_tr, 'r-', label=("Number of Transactions (#)\n" + str(list_blockchain_time[0]) + "\n" + str(list_blockchain_time[1])))
        plt.plot(x_vals_size, 'go', label="Block Size (kb)")
        plt.legend(loc="best")
        plt.xlabel("block number")
        axes.set_xlim([0, len(x_vals_size)])

        plt.savefig('plot/' + description + '(' + str(len(x_vals_size)) + ')')
        print("plot " + description + ".png created")
    # -----------------------------------------------------------------------------------------------------------------------------
    # ------------------------- TO IMPLEMENT THROUGHPUT -------------------------
    elif(description == "tthroughput"):
        x_vals = get_list_from_file("creation_time")
        x_vals[:] = [float(x) for x in x_vals]

        y_vals = get_list_from_file("transactions")
        y_vals[:] = (float(x) for x in y_vals)

        to_plot = []
        to_plot[:] = [y/(x+1) for x,y in zip(x_vals, y_vals)]
        to_plot.sort()

        hist, bins = np.histogram(to_plot, bins=1000)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, align='center', width=width, label="Throughput tr/s\n" + str(list_blockchain_time[0]) + "\n" + str(list_blockchain_time[1]), alpha=0.5, facecolor='green')
        plt.legend(loc="best")

        axes.set_xlim([0,50])
        plt.xlabel("transactions/second")
        plt.ylabel("blocks (" + str(len(x_vals)) + " retrieved)")

        plt.savefig('plot/' + description + '(' + str(len(x_vals)) + ')')

    # -----------------------------------------------------------------------------------------------------------------------------
    elif(description == "fee_bandwidth"):
        epoch_vals, x_vals, y_vals = get_lists_ordered("epoch", "creation_time", "fee")

        epoch_vals[:] = [int(x) for x in epoch_vals]

        x_vals[:] = [float(x) for x in x_vals]
        x_vals[:] = [x / 60 for x in x_vals] # in minutes

        y_vals[:] = [float(y) for y in y_vals]
        y_vals[:] = [y / 100000000 for y in y_vals] # in BTC

        regression = []
        regression.append("$f_{Bg}(x)$")
        regression.append(2)
        plot_multiple_lists(description, marker_list[0], epoch_vals, x_vals, y_vals, regression=regression)

        plt.ylabel("fee $(BTC)$")
        plt.xlabel("creation time (min)")
        axes.set_xlim([0, 30])
        axes.set_ylim([0, 0.5])

        plt.savefig('plot/' + description + '(' + str(len(x_vals)) + ')')
        print("plot " + description + ".png created")
    elif(description == "fee_transactions"):
        y_vals = get_list_from_file("avgttime")
        y_vals[:] = [float(x) for x in y_vals]
        y_vals[:] = [x / 60 for x in y_vals] # in minutes

        x_vals = get_list_from_file("fee")
        x_vals[:] = [float(x) for x in x_vals]
        x_vals[:] = [x / 100000000 for x in x_vals]  # in BTC

        # divide the average fee paid fot the number of transaction in that block
        num_tr = get_list_from_file("transactions")
        num_tr[:] = [float(x) for x in num_tr]
        x_vals[:] = [x / y for x,y in zip(x_vals, num_tr)]

        plt.plot(x_vals, y_vals, 'ro', label=(
            "transaction visibility\n" + str(list_blockchain_time[0]) + "\n" + str(list_blockchain_time[1])))
        plt.xlabel("$\overline{T_p}$ (BTC)")
        plt.ylabel("transaction visibility (min)")
        axes.set_ylim([0, max(y_vals)/10])
        axes.set_xlim([0, 0.006])
        if (regression):
            model = np.polyfit(x_vals, y_vals, 1)
            x_vals.sort()
            predicted = np.polyval(model, x_vals)
            plt.plot(x_vals, predicted, 'g-', label="regression", lw=4)
            polynomial = np.polyfit(x_vals, predicted, 2)
            print polynomial

        plt.legend(loc="best")
        plt.savefig('plot/' + description + '(' + str(len(x_vals)) + ')')
        print("plot " + description + ".png created")

def check_blockchain():
    """
    check if the element in the local blockchain have plausible datz, if not, local blockchain is not in a good status,
    in that case is better to create a new file.
     - check whether the block size is in between 1kb and 2 MB

    :return: True or False
    """
    check = True
    if (os.path.isfile(file_blockchain)):
        list = get_list_from_file("size")
        for i in list:
            if ((int(i) > 2000000) or (int(i) < 100)):
                check = False
            if (len(list) % n_portions != 0):
                check = False
    return check

def get_number_blocks():
    """
    :return: number of the current blocks saved in the local blockchain - 0 if file doesn't exist
    """
    number = 0
    if (os.path.isfile(file_blockchain)):
        hash_list = get_list_from_file("hash")
        number = len(hash_list)
    return number

def create_interval_lists():
    """
    create lists with integer containing the intervals in the info.txt file
    :return: start height list and end height list
    """
    start_list = []
    end_list = []
    if (os.path.isfile(file_info)):
        with open(file_info, "r") as file:
            file.seek(0)
            file_lines = file.readlines()

            for line in file_lines:
                new_line = line.split("--")

                start_list.append([int(s) for s in new_line[0].split() if s.isdigit()][0])
                end_list.append([int(s) for s in new_line[1].split() if s.isdigit()][0])
    return start_list, end_list

def get_earliest_hash():
    """
    if exists, get the earliest hash saved in the blockchain local file
    :return: the earliest hash in the local blockchain file - empty string if file doesn't exist
    """
    earliest_hash = ""
    if (os.path.isfile(file_blockchain)):
        hash_list = get_list_from_file("hash")
        if (hash_list != False):
            earliest_hash = hash_list[-1]
        else:
            earliest_hash = False
    return earliest_hash


def epoch_datetime(epoch):
    """
    convert epoch to datetime %Y-%m-%d %H:%M:%S
    :param epoch: time in epoch
    :return: time in datetime with %Y-%m-%d %H:%M:%S format
    """
    datetime = time.strftime('%d-%m-%Y %H:%M:%S', time.localtime(float(epoch)))
    return datetime

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

def check_hash(startH, endH):
    """
    Check whether the interval of height that needs to be retrieved are already retrieved in the file .txt
    :param startH: start height
    :param endH: end height
    :return:   - True  : if is not possible to retrieve the sequence required
               - False : if the sequence is not in the file so is possible to retrieve it
    """
    toReturn = False
    height_list = get_list_from_file("height")
    for h in height_list:
        if(int(startH) <= int(h) <= int(endH)):
            toReturn = True
    return toReturn

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


def blockchain_info():
    """
    print the information regarding the local blockcahin
    :return: string containing the info from the blockchain text file
    """
    string_return = ""
    if (os.path.isfile(file_blockchain)):
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
    with io.FileIO(file_info, "w+") as file:
        file.write(interval_string)

    return interval_string


def copyFiles(source, destination):
    """
    copies the source file in destination to have a backup
    :param source       :   file to copy
    :param destination  :   destination file
    """
    if (os.path.isfile(source)):
        copyfile(source, destination)
        print ("file " + source + " copied in " + destination)

def delete_file(file):
    """
    delete a file from the directory
    :param file: str, name of the file to be deleted

    """
    try:
        os.remove(file)
        print file + " deleted."
    except OSError:
        pass

def myComplexFunc(x, a, b, c):
    return a * np.power(x, b) + c

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

def fBg(x):
    """
    Function defined in the paper that explains the relation between the fee paid to the miners and the block
    creation time
    :param x : creation time
    :return : the expected fee according to the function generated from the regression
    """
    y = - ((1/(10**4))*(x**2)) + ((3/(10**2))*(x)) + 0.3
    return y


def percentage(part, whole):
  return 100 * float(part)/float(whole)

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
