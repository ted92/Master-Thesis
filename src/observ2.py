# -*- coding: utf-8 -*-
"""ete011 @Enrico Tedeschi - UiT v2
Observing the Bitcoin blockchain in real time. The system will retreive portion of the Bitcoin blockchain,
do data analysis, generating models and plotting the results.

v2: the order of block retrieval doesn't matter, is possible to retrieve blocks starting from a certain hash

Usage: observ.py -t number
    -h | --help         : usage
    -i                  : gives info of the blockchain retrieved
    -t number           : adds on top a number of blocks. The blocks retreived will be the most recent ones. If the blockchain growth more than the block requested do -u (update)
    -e number           : appends blocks at the end of the .txt file. Fetch older blocks starting from the last retrieved
    -P                  : plots all
    -p start [end]      : plots data in .txt file in a certain period of time, from start to end. If only start then consider from start to the end of the .txt file
    -R                  : plots the regression and the models that predict the blockchain
    -r start [end]      : plots the regression and the models in a certain period of time, from start to end. If only start then consider from start to the end of the .txt file
    -u                  : updates the local blockchain to the last block created
    -c start end        : retrieves blocks to compare the blockchain in different epoch. The height to be retrieved is given from start and end

"""

import sys, getopt
from docopt import docopt
import io
from blockchain import blockexplorer
import numpy as np
import string
import re
import os.path
import matplotlib.pyplot as plt
import datetime
import time
from time import sleep
import urllib2
import matplotlib.patches as mpatches
from forex_python.converter import CurrencyRates
from forex_python.bitcoin import BtcConverter
from scipy.optimize import curve_fit
import statsmodels.api as sm
import matplotlib.lines as mlines
import matplotlib.axis as ax
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
import json

# ------ GLOBAL ------
global file_blockchain
file_blockchain = "blockchain.txt"

global file_info
file_info = "info.txt"

global n_portions
n_portions = 4

global color_list
color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

global marker_list
marker_list = ['o', '-', '*', '^']

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

        opts, args = getopt.getopt(argv, "hiuRPc:p:t:e:r:")
        valid_args = False

        for opt, arg in opts:
            if(opt == "-t"):    # update on top
                print ("Adding at the top the latest " + arg + " blocks.")
                number_of_blocks = int(arg)
                get_blockchain(number_of_blocks)
                valid_args = True
            if(opt == "-e"):    # append blocks at the end
                print ("Appending at the end " + arg + " blocks.")
                number_of_blocks = int(arg)
                get_blockchain(number_of_blocks, earliest_hash)
                valid_args = True
            if(opt == "-u"):    # update with the missing blocks
                str_update = update_blockchain()
                valid_args = True
                if (str_update != None):
                    print str_update
            if(opt == "-i"):    # blockchain info
                print blockchain_info()
                valid_args = True
            if(opt == "-h"):    # usage
                print (__doc__)
                valid_args = True
            if(opt == "-P"):    # plot only
                plot_sequence(False, start_v, end_v)
                valid_args = True
            if(opt == "-p"):    # plot
                end_v = int(arg)
                if(args):
                    start_v = int(args[0])
                plot_sequence(False, start_v, end_v)
                valid_args = True
            if (opt == "-R"):  # regression only
                plot_sequence(True, start_v, end_v)
                valid_args = True
            if (opt == "-r"):  # plot regression with start and end
                end_v = int(arg)
                if (args):
                    start_v = int(args[0])
                plot_sequence(True, start_v, end_v)
                valid_args = True
            if(opt == "-c"):    # compare with older blocks - retrieve blockchain in previous time
                blocks = int(arg)

                # check wheter if a portion of the interval to be retrieved is already in the file
                """in_file = check_hash(start_height, end_height)
                if(in_file == True):
                    print (bcolors.WARNING + "Warning: " + bcolors.ENDC +"The protion you are trying to retrieve is already present in the " + file_blockchain)
                elif(in_file == False):
                    print ("Retreiving " + str(blocks) + " blocks starting from: " + s_time)
                    # todo: re-do the get_blockchain with an interval one

                    # get_blockchain(blocks, hash)"""
                define_intervals(blocks)
                valid_args = True

        if(valid_args == False):
            print (__doc__)
    except getopt.GetoptError:
        print (__doc__)
        sys.exit(2)


def define_intervals(number_of_blocks):
    """
    Retrieves blocks with a certain interval according to how many blocks are present in the blockchain.
    These intervals are called portions.
    We set a number of portions = 3
    :param number_of_blocks:
    :param hash:
    :return:
    """
    error = False
    # define p = number of blocks per portion
    # n = number of blocks in the blockchain

    last_block = blockexplorer.get_latest_block()
    n = last_block.height
    p = n / n_portions

    start_list, end_list = create_interval_lists()
    i = 0
    # case if the files doesn't exist
    if (start_list == []):
        # starting from 0


        # get the heights and hashes where to start:
        while (i < n_portions):
            try:
                if(i == 0):
                    # the retrieval starts with the latest block, and then the previous are retrieved as well
                    height_to_start = 1 + number_of_blocks
                    b_array = blockexplorer.get_block_height(height_to_start)
                else:
                    height_to_start = i*p + (number_of_blocks)
                    b_array = blockexplorer.get_block_height(height_to_start)
                b = b_array[0]
                hash = b.hash
                epoch = b.time
                time = epoch_datetime(epoch)
                i += 1
                print "Retrieving " + str(number_of_blocks) + " starting from " + time
                get_blockchain(number_of_blocks, error, hash)
            except Exception as e:
                error = True
                json_req = urllib2.urlopen(
                    "https://blockchain.info/block-height/" + str(height_to_start) + "?format=json").read()
                blocks_retrieved = json.loads(json_req)
                b = blocks_retrieved["blocks"]
                block = b[0]
                height = int(block['height'])
                i += 1
                get_blockchain(number_of_blocks, error, height)

    # case if the files exists already
    else:
        # in start_list and end_list are stored all the heights representing the intervals retrieved
        # todo: control that the number of blocks is not higher than the portion p
        if (int(end_list[0]) + number_of_blocks >= p):
            print (bcolors.WARNING + "WARNING: " + bcolors.ENDC + "Blockchain already up to date!")
        else:
            while (i < n_portions):
                try:
                    height_to_start = end_list[i] + (number_of_blocks) + 1
                    b_array = blockexplorer.get_block_height(height_to_start)
                    b = b_array[0]
                    hash = b.hash
                    epoch = b.time
                    time = epoch_datetime(epoch)
                    i += 1
                    print "Retrieving " + str(number_of_blocks) + " blocks starting from " + time
                    get_blockchain(number_of_blocks, error, hash)
                except Exception as e:
                    error = True
                    json_req = urllib2.urlopen(
                        "https://blockchain.info/block-height/" + str(height_to_start) + "?format=json").read()
                    blocks_retrieved = json.loads(json_req)
                    b = blocks_retrieved["blocks"]
                    block = b[0]
                    height = int(block['height'])
                    i += 1
                    get_blockchain(number_of_blocks, error, height)

def plot_sequence(regression,  start_v, end_v):
    """
    @params:
      bool regression: if "-r" or "-R" is True, false otherwise
      int start_v: start value where the blockchain will be plotted
      int end_v: end value where the blockchain will be plotted
    :return:
    """
    if(regression):
        plot_data("growth_blockchain", 2, True, start=start_v, end=end_v)
        plot_data("fee_bandwidth", 3, True, start=start_v, end=end_v)
        plot_data("fee_transactions", 7, True, start=start_v, end=end_v)
    else:
        plot_data("time_per_block", 0, start=start_v, end=end_v)
        plot_data("byte_per_block", 1, start=start_v, end=end_v)
        """plot_data("growth_blockchain", 2, start=start_v, end=end_v)
        plot_data("fee_bandwidth", 3, start=start_v, end=end_v)"""
        plot_data("bandwidth", 4, start=start_v, end=end_v)
        """plot_data("efficiency", 5, start=start_v, end=end_v)
        plot_data("transaction_visibility", 6, start=start_v, end=end_v)
        plot_data("fee_transactions", 7, start=start_v, end=end_v)
        plot_data("tthroughput", 8, start=start_v, end=end_v)"""

# @profile
def get_blockchain(number_of_blocks, error, hash = None):
    """
    it retreives blocks from blockchain starting from the last block if hash is none,
    otherwise start from the block hash given in input
    @params:
     int number_of_blocks: blocks to retrieve
     bool error: if True data are retrieved in Json if False through the client API
     str hash: hash of the block from where to start the retrieval
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
    printProgress(index_progress_bar, number_of_blocks, prefix='Saving Blockchain:', suffix='Complete',
                  barLength=50)
    # ---------------------------------
    try:
        last_block = blockexplorer.get_block(hash)
        start_time = datetime.datetime.now()
        current_block = blockexplorer.get_block(last_block.previous_block)
        end_time = datetime.datetime.now()

    except Exception as e:
        error = True
        start_time = datetime.datetime.now()
        json_req = urllib2.urlopen(
            "https://blockchain.info/block-index/" + str(hash) + "?format=json").read()
        end_time = datetime.datetime.now()
        current_block = json.loads(json_req)

    for i in range(number_of_blocks):
        try:
            # ---------- PROGRESS BAR -----------
            sleep(0.01)
            index_progress_bar += 1
            printProgress(index_progress_bar, number_of_blocks, prefix='Saving Blockchain:', suffix='Complete', barLength=50)
            # -----------------------------------

            time_to_fetch = end_time - start_time
            time_in_seconds = get_time_in_seconds(time_to_fetch)
            fetch_time_list.append(time_in_seconds)

            start_list, end_list = create_interval_lists()

            if(error == False):
                # ---- List creation
                epoch = current_block.time
                epoch_list.append(epoch)

                hash = current_block.hash
                hash_list.append(hash)

                fee = current_block.fee
                fee_list.append(fee)

                size = current_block.size
                size_list.append(size)

                height = current_block.height
                height_list.append(height)

                avg_tr = get_avg_transaction_time(current_block, False)
                avg_transaction_list.append(avg_tr)

                block_size = float(size) / 1000000 # -------> calculate read Bandwidth with MB/s
                bandwidth = block_size / time_in_seconds
                bandwidth_list.append(bandwidth)

                transactions = current_block.transactions
                list_transactions.append(len(transactions))

                miner = current_block.relayed_by
                list_miners.append(miner)

                received_time = current_block.received_time
                list_received_time.append(received_time)

                # --- creation time list--
                start_time = datetime.datetime.now() # -------------------------------------------------------------------------
                prev_block = blockexplorer.get_block(current_block.previous_block)
                end_time = datetime.datetime.now()  # --------------------------------------------------------------------------
                prev_epoch_time = prev_block.time
                current_creation_time = current_block.time - prev_epoch_time
                # -- check if the creation time is negative
                """if (current_creation_time < 0):
                    current_creation_time = get_creation_time(current_block, prev_block, False, False, current_creation_time)"""
                # -------------------------------------------
                creation_time_list.append(current_creation_time)
                # ------------------------

                add_mining_nodes(current_block)

                current_block = prev_block

            else:
                epoch = current_block["time"]
                epoch_list.append(epoch)

                hash = current_block["hash"]
                hash_list.append(hash)

                fee = current_block["fee"]
                fee_list.append(fee)

                size = current_block["size"]
                size_list.append(size)

                height = current_block["height"]
                height_list.append(height)

                avg_tr = get_avg_transaction_time(current_block, True)
                avg_transaction_list.append(avg_tr)

                block_size = float(size) / 1000000  # -------> calculate read Bandwidth with MB/s
                bandwidth = block_size / time_in_seconds
                bandwidth_list.append(bandwidth)

                transactions = len(current_block["tx"])
                list_transactions.append(transactions)

                miner = current_block["relayed_by"]
                list_miners.append(miner)

                received_time = current_block["received_time"]
                list_received_time.append(received_time)


                prev_block = current_block["prev_block"]
                start_time = datetime.datetime.now()  # ------------------------------------------------------------------------
                prev_block = blockexplorer.get_block(prev_block)
                end_time = datetime.datetime.now()  # --------------------------------------------------------------------------

                prev_epoch_time = prev_block.time
                current_creation_time = current_block["time"] - prev_epoch_time
                """if (current_creation_time < 0):
                    current_creation_time = get_creation_time(current_block, prev_block, True, True, current_creation_time)"""
                creation_time_list.append(current_creation_time)

                # add_mining_nodes(current_block)

                current_block = prev_block

                error = False
        except Exception as e:
            start_time = datetime.datetime.now()  # -------------------------------------------------------------------------
            if error:
                json_req = urllib2.urlopen(
                    "https://blockchain.info/block-index/" + current_block["prev_block"] + "?format=json").read()
                print "\nhere in true"
                print e
            else:
                json_req = urllib2.urlopen(
                "https://blockchain.info/block-index/" + current_block.previous_block + "?format=json").read()
                print "\nhere in false"
                print e
            end_time = datetime.datetime.now()  # ---------------------------------------------------------------------------

            prev_block = json.loads(json_req)
            prev_epoch_time = prev_block["time"]

            if error:
                current_creation_time = current_block["time"] - prev_epoch_time
                """if(current_creation_time < 0):
                    current_creation_time = get_creation_time(current_block, prev_block, True, True, current_creation_time)"""
            else:
                current_creation_time = current_block.time - prev_epoch_time
                """if (current_creation_time < 0):
                    current_creation_time = get_creation_time(current_block, prev_block, True, False, current_creation_time)"""
            creation_time_list.append(current_creation_time)

            # add_mining_nodes(current_block)

            current_block = prev_block

            error = True

        """except KeyError:
            # retreive block from the website
            # -- URL
            start_time = datetime.datetime.now()  # -------------------------------------------------------------------------
            json_req = urllib2.urlopen(
                "https://blockchain.info/block-index/" + current_block.previous_block + "?format=json").read()
            end_time = datetime.datetime.now()  # ---------------------------------------------------------------------------

            prev_block = json.loads(json_req)
            prev_epoch_time = prev_block["time"]
            current_creation_time = current_block.time - prev_epoch_time
            creation_time_list.append(current_creation_time)

            # add_mining_nodes(current_block)

            current_block = prev_block

            error = True"""
            # ------

    to_write_list = [hash_list, epoch_list, creation_time_list, size_list, fee_list, height_list, bandwidth_list, list_transactions, avg_transaction_list, list_miners, list_received_time]
    # to_write_list[9] = mining_list
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
    printProgress(index_progress_bar, n, prefix='Writing .txt file:', suffix='Complete',
                  barLength=50)

    with io.FileIO(file_blockchain, "a+") as file:
        for i in range(n):
            # --- WRITE IN FILE ---
            write_file(to_write_list, file, i)
            # ---------------------

            # -------- PROGRESS BAR -----------
            sleep(0.01)
            index_progress_bar += 1
            printProgress(index_progress_bar, n, prefix='Writing .txt file:', suffix='Complete',
                          barLength=50)
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
            json_req = urllib2.urlopen(
                "https://blockchain.info/block-index/" + prevblock["prev_block"] + "?format=json").read()
            right_ancestor = json.loads(json_req)
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
    :return: list containig the growing time
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
        block_time = float(block["time"])
        tx = block["tx"]

        t_sum = 0
        for t in tx:
            approval_time = block_time - float(t["time"])
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


def get_indexes():
    """
    get the indexed where the list needs to be splitted
    :return : list with indexes
    """
    start_list, end_list = create_interval_lists()
    interval_list_start = []
    interval_list_end = []

    interval = end_list[0]

    i = 0
    while (i < n_portions):
        if (i == 0):
            interval_list_start.append(0)
            interval_list_end.append(interval)
        else:
            interval_list_start.append((i*interval)+1)
            interval_list_end.append(interval*(i+1))
        i += 1

    return interval_list_start, interval_list_end

def plot_multiple_lists(description, marker, list1, list2, list3 = None):
    """
    method that take care of plotting the data.
    :param description  - Required  :   description of the graph to plot
    :param marker       - Required  :   marker to plot, could be '-', 'o', '*' ecc...
    :param list1        - Required  :   list with x values to plot
    :param list2        - Required  :   list with epoch to put on lable
    :param list3        - Optional  :   second list with the y values to plot
    """
    if(list3 != None):
        is_x = True
    else:
        is_x = False

    # retrieve the intervals to print
    start_interval, end_interval = get_indexes()

    # plot the portions
    to_plot_1 = []
    to_plot_2 = []
    to_plot_3 = []

    # create data to plot having a list of a list in to_plot_x and to_plot_y
    i = 0
    while (i < n_portions):
        to_plot_1.append(list1[start_interval[i]:end_interval[i]])
        to_plot_2.append(list2[start_interval[i]:end_interval[i]])

        if(is_x):
            to_plot_3.append(list3[start_interval[i]:end_interval[i]])
            plt.plot(to_plot_2[i], to_plot_3[i], color_list[i] + marker,
                     label=(str(epoch_datetime(to_plot_1[i][0])) + "\n" + str(epoch_datetime(to_plot_1[i][-1]))))
        else:
            plt.plot(to_plot_2[i], color_list[i] + marker,
                 label=(str(epoch_datetime(to_plot_1[i][0])) + "\n" + str(epoch_datetime(to_plot_1[i][-1]))))
        i += 1


def plot_data(description, plot_number, regression = None, start = None, end = None):
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
    :param regression   - Optional  : write a regression on the plot generated
    :param start        - Optional  : block number where the plot starts
    :param end          - Optional  : block number where the plot ends
    """
    list_blockchain_time = datetime_retrieved(start, end)
    plt.figure(plot_number)
    plt.rc('lines', linewidth=1)
    axes = plt.gca()

    if(description == "time_per_block"): # shows the creation time for each block
        y_vals, x_vals = get_lists_ordered("creation_time", "epoch")
        x_vals[:] = [float(x) for x in x_vals]
        y_vals[:] = [float(y) for y in y_vals]

        y_vals[:] = [y / 60 for y in y_vals]

        plot_multiple_lists(description, marker_list[0], x_vals, y_vals)

        plt.legend(loc="best")
        plt.ylabel("creation time (min)")
        plt.xlabel("epoch")
        axes.set_ylim([0, 40])

        plt.savefig('plot/' + description + '(' + str(len(x_vals)) + ')')

        print("plot " + description + ".png created")
    elif(description == "byte_per_block"): # shows the size of the blocks during time
        y_vals, x_vals = get_lists_ordered("size", "epoch")
        y_vals[:] = [float(i) for i in y_vals]
        y_vals[:] = [y / 1000000 for y in y_vals]
        y_vals = y_vals[end:start]

        plot_multiple_lists(description, marker_list[0], x_vals, y_vals)

        plt.legend(loc="best")
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
        y_vals = y_vals[end:start]

        plot_multiple_lists(description, marker_list[1], x_vals, y_vals)

        # plt.plot(x_vals, 'c-', label=(
            # "read bandwidth Mb/s\n" + str(list_blockchain_time[0]) + "\n" + str(list_blockchain_time[1])), lw=3)

        plt.legend(loc="best")
        plt.ylabel("read bandwidth (Mb/s)")
        plt.xlabel("block number")
        axes.set_xlim([0, len(y_vals)/n_portions])
        max_in_list = max(y_vals)
        axes.set_ylim([0, max_in_list * 1.2])

        plt.savefig('plot/' + description + '(' + str(len(x_vals)) + ')')
        print("plot " + description + ".png created")
    elif(description == "growth_blockchain"):
        time_list = get_list_from_file("creation_time")
        time_list[:] = [float(x) for x in time_list]

        size_list = get_list_from_file("size")
        size_list[:] = [float(x) for x in size_list]

        size_list = size_list[end:start]
        time_list = time_list[end:start]

        x_vals = create_growing_time_list(time_list)
        y_vals = create_growing_size_list(size_list)

        """# ---- get the exact data
        elements = len(y_vals)
        last_size = float(y_vals[elements-1])

        last_size = last_size/1000000
        print last_size

        elements = len(x_vals)
        last_time = float(x_vals[elements-1])
        print  last_time

        # ------------"""
        x_vals[:] = [float(x) for x in x_vals]
        x_vals[:] = [x / 60*60 for x in x_vals] # in hours

        y_vals[:] = [float(y) for y in y_vals]
        y_vals[:] = [y / 1000000000 for y in y_vals] # in GB

        plt.ylabel("size (GB)")
        plt.xlabel("time (h)")
        # axes.set_xlim([0, max(x_vals)])

        if(regression):
            el = len(x_vals)
            last_el = x_vals[el - 1]

            # ---- get the predicted date time --------
            epoch_list = get_list_from_file("epoch")
            epoch_list = epoch_list[end:start]
            last_epoch = int(epoch_list[0])
            # add the hours to that epoch
            sec_to_add = (last_el*3) * 60 * 60
            last_epoch = last_epoch + sec_to_add
            prediction_date = epoch_datetime(last_epoch)
            # -----------------------------------------

            newX = np.linspace(0, last_el * 3)
            popt, pcov = curve_fit(myComplexFunc, x_vals, y_vals)
            plt.plot(newX, myComplexFunc(newX, *popt), 'g-', label=("prediction until\n" + str(prediction_date)), lw=3)
            lim = axes.get_ylim()
            axes.set_ylim([0, lim[1]])
            polynomial = np.polyfit(newX, myComplexFunc(newX, *popt), 2)
            print polynomial

        plt.plot(x_vals, y_vals, 'ro', label=(
            "growth retrieved\n" + str(list_blockchain_time[0]) + "\n" + str(list_blockchain_time[1])),
                 markevery=(len(x_vals) + 100) / 100)
        plt.legend(loc="best")

        plt.savefig('plot/' + description + '(' + str(len(x_vals)) + ')')
        print("plot " + description + ".png created")
    elif(description == "transaction_visibility"):
        x_vals = get_list_from_file("avgttime")
        x_vals[:] = [float(x) for x in x_vals]
        x_vals[:] = [x / 60 for x in x_vals]    # in minutes
        x_vals = x_vals[end:start]
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

        x_vals_size = x_vals_size[end:start]
        x_vals_time = x_vals_time[end:start]
        x_vals_tr = x_vals_tr[end:start]

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
        x_vals[:] = x_vals[end:start]
        x_vals[:] = [float(x) for x in x_vals]

        y_vals = get_list_from_file("transactions")
        y_vals[:] = y_vals[end:start]
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
        x_vals = get_list_from_file("creation_time")
        x_vals[:] = [float(x) for x in x_vals]
        x_vals[:] = [x / 60 for x in x_vals] # in minutes

        y_vals = get_list_from_file("fee")
        y_vals[:] = [float(x) for x in y_vals]
        y_vals[:] = [x / 100000000 for x in y_vals] # in BTC

        x_vals = x_vals[end:start]
        y_vals = y_vals[end:start]

        plt.plot(x_vals, y_vals, 'ro', label=(
            "fee paid\n" + str(list_blockchain_time[0]) + "\n" + str(list_blockchain_time[1])))
        plt.ylabel("fee (BTC)")
        plt.xlabel("creation time (min)")
        axes.set_xlim([0, max(x_vals)])
        axes.set_ylim([0, 2.5])

        if(regression):
            # logarithmic regression
            newX = np.logspace(0, 2, base=10)
            popt, pcov = curve_fit(myComplexFunc, x_vals, y_vals)
            plt.plot(newX, myComplexFunc(newX, *popt), 'g-', label="regression", lw=5)
            polynomial = np.polyfit(newX, myComplexFunc(newX, *popt), 2)
            print polynomial

        plt.legend(loc="best")
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

        x_vals = x_vals[end:start]
        y_vals = y_vals[end:start]

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
        list2 = [x[3] for x in sorted_together]
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
                            "Blockchain contains errors. Throw the file and make a new one")

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


def update_blockchain():
    """
    update the local blockchain retrieving the latest blocks that are missing
    :return: string with the status
    """
    string_return = None
    error = False
    if (os.path.isfile(file_blockchain)):
        # count how many nodes are missing
        height = get_list_from_file("height")
        last_retreived = int(height[0])
        current_block = blockexplorer.get_latest_block()
        last_total = int(current_block.height)
        diff = last_total - last_retreived
        if (diff > 0):
            print ("Updating the blockchain (" + str(diff) + " blocks missing)...")
            diff = diff + 1
            get_blockchain(diff, error)
        else:
            print ("Blockchain already up to date!")
    else:
        string_return = "File still doesn't exist. You need to fetch blocks first with -t command.\n" + str(__doc__)
    return string_return


def myComplexFunc(x, a, b, c):
    return a * np.power(x, b) + c

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
