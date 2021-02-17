#! /usr/bin/env python
# coding: utf8

import pandas as pd
import sqlite3
import copy
import time
import math
import random
from datetime import datetime,date
import csv
import sys
import argparse

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('input_file_name')
    parser.add_argument('num_iter',type=int)
    parser.add_argument('output_file_name')
    parser.add_argument('--teleport','-d',type=float)
    parser.add_argument('--laziness','-q',type=float)
    parser.add_argument('--convergence',type=bool)
    parser.add_argument('--version',type=int)
    parser.add_argument('--verbose',type=bool)
    parser.add_argument('--date',type=bool)

    args = parser.parse_args()

    date_format = "%Y-%m-%d"

    if args.teleport:
        d = args.teleport
    else:
        d=0.01

    if args.laziness:
        q = args.laziness
    else:
        q = 0

    if args.convergence:
        convergence = True
    else:
        convergence = False

    if args.version:
        version = args.version
    else:
        version = 2

    if args.verbose:
        verbose = True
    else:
        verbose = False

    if args.date:
        date_input = True
    else:
        date_input = False

    if verbose:
        print("Reading input file...")
        tStart = time.time()

    jour_s_d = {}
    ensemble_noeuds = set()
    nbRow = 0
    reader = csv.reader(open(args.input_file_name, 'rt'))

    for row in reader:
        if date_input:
            current_date = datetime.strptime(row[0])
        else:
            current_date = int(row[0])

        if current_date in jour_s_d.keys():
            if row[1] in jour_s_d[current_date].keys():
                jour_s_d[current_date][row[1]].append(row[2])
            else:
                jour_s_d[current_date][row[1]] = [row[2]]
        else:
            jour_s_d[current_date] = {row[1] : [row[2]]}
        ensemble_noeuds.add(row[1])
        ensemble_noeuds.add(row[2])

        nbRow += 1
        if nbRow % 100000 == 0:
            print(str(nbRow) + ' movements processed')

    all_nodes=tuple(ensemble_noeuds)

    if verbose:
        tEnd = time.time()
        print('Input file read.')
        print('Input file reading time: ' + str(round(tEnd - tStart, 2)) +
            'seconds.')

        print('Number of nodes: ' + str(len(ensemble_noeuds)))

        print("Starting random walks...")
        tStart = time.time()

    if convergence==True:
        starting_dict = {}
        for i in range(100):
            starting_dict = marches_aleatoires(
                jour_s_d,all_nodes,
                int(args.num_iter/100),q,d,starting_dict,version,date_input
                )
            current_file_name= args.output_file_name + "_" +str(i) + ".csv"
            with open(current_file_name, 'w',newline='') as csv_file:
                writer = csv.writer(csv_file,delimiter=',')
                for line in starting_dict.items():
                    temp = line + (str(q),str(d),i*args.num_iter/100)
                    writer.writerow(temp)
    else:
        dict_nombre_passages = marches_aleatoires(jour_s_d,
        all_nodes,args.num_iter,q,d,{},version
        )
        with open(args.output_file_name, 'w',newline='') as csv_file:
            writer = csv.writer(csv_file,delimiter=',')
            for line in dict_nombre_passages.items():
                temp = line + (str(q),str(d),args.num_iter)
                writer.writerow(temp)

    if verbose:
        tEnd = time.time()
        print('Random walk simulation time: ' + str(round(tEnd - tStart, 2)) +
            'seconds.')

def marches_aleatoires(jour_s_d,all_nodes,
    num_iter,q,d,starting_dict,version,date_input):
    nb_pass = starting_dict

    starting_time = random.choice(list(jour_s_d.keys()))
    i = starting_time
    min_time=min(jour_s_d)
    max_time=max(jour_s_d)
    state = random.choice(all_nodes)

    k=0

    while (k < num_iter):
        if i <= max_time:
            if(version==1): # In version 1, presence is logged
                nb_pass[state] = nb_pass.get(state,0) + 1
            if state in jour_s_d[i].keys():
                paresse = random.random()
                if(paresse > q):
                    teleportation = random.random()
                    if(teleportation < d): # Teleportation with probability d
                        state = random.choice(all_nodes)
                    else: # Follow a random edge with probability 1-d
                        if(version==2): # In version 2, outgoing edge movements
                        # are logged
                            nb_pass[state] = nb_pass.get(state,0) + 1
                        potential = jour_s_d[i][state]
                        state=random.choice(potential)
            else: # Case where state has no neighbors at time t
                paresse = random.random()
                if(paresse > q):
                    teleportation = random.random()
                    if(teleportation < d): # Teleport with probability d
                        state = random.choice(all_nodes)
            if date_input:
                i = i + datetime.timedelta(days=1)
            else:
                i+=1
            k+=1
        else: # Go back to first time
            i = min_time
        if k % 1000000==0:
            if verbose: print('Number of steps: '+ str(k)+'.')
    return nb_pass

if __name__ == '__main__':
    main()
