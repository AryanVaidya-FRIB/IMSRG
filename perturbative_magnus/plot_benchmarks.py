#!/usr/bin/env python

#------------------------------------------------------------------------------
# plot_benchmarks.py
#
# author:   A. Vaidya
# version:  1.0
# date:     Jan 22, 2025
# 
# tested with Python v3.10
# 
# Plots results from multiple benchmark files for the P3H model
# iterating through values of g, for fixed value of b. 
# Plots GS energy vs. g, num steps vs. g, time per flow vs. g, 
# and RAM use vs. g. Needs to take 2 inputs, delta and b
#
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# Plotting Routine
#------------------------------------------------------------------------------
def plot_values(j, indexList, dataList, labels, metric, title, saveName):
    plt.figure()
    for i in indexList:
        plt.plot(dataList[i].iloc[:,0].to_numpy(), dataList[i].iloc[:,j].to_numpy(), label = labels[i])
    
    plt.legend()
    plt.xlabel("g")
    plt.ylabel(metric)
    plt.title(title)
    plt.savefig(saveName)
    return

#------------------------------------------------------------------------------
# Main Program
#------------------------------------------------------------------------------

def main():
    delta = "1.0"
    b     = "+0.4828"

    # Files to pull
    dPath      = "/mnt/c/Users/aryan/Documents/MSU_FRIB/IMSRG/perturbative_magnus/"
    filePath   = dPath + "benchmarks/"
    outPath    = dPath + "plots/"
    fileString = f"imsrg-white_d{delta}_b{b}_N4_"
    
    # List of each benchmark
    flows = ["magnus","ev1", "perturbativeStored", "perturbative2Stored", "perturbativeBCH", "perturbative2BCH"]
    labels = ["Magnus","Direct Flow", "pMagnus Stored", "pMagnus2 Stored", "pMagnus BCH", "pMagnus2 BCH"]
    indexList = range(len(flows))

    noMagnus = indexList[1:]
    order_comparison = indexList[:4]

    # Load datafiles as a list of Pandas Dataframes
    dataList = []
    for i in indexList:
        dataList.append(pd.read_csv(filePath+fileString+flows[i]+".csv", index_col=0))

    onlyFirstOrder = [0,1,4,5]
    plot_values(1, onlyFirstOrder, dataList, labels, "GS Energy", f"GS Energy vs. g - b={b}", outPath+f"GSEnergy_b{b}_pMagBCH.jpg")
    plot_values(2, onlyFirstOrder, dataList, labels, "s", f"s vs. g - b={b}", outPath+f"num_steps_b{b}_pMag.jpg")
    plot_values(3, onlyFirstOrder, dataList, labels, "Time Spent (seconds)", f"Time per Flow vs. g - b={b}", outPath+f"time_b{b}_pMagBCH.jpg")
    plot_values(4, onlyFirstOrder, dataList, labels, "RAM (kb)", f"Memory Usage vs. g - b={b}", outPath+f"RAM_b{b}_pMagBCH.jpg")
    return
    # Plot GS energy vs. g - index 1
    # 3 plots generated - 1 for all of them together, 1 for without magnus, 1 without BCH terms
    plot_values(1, indexList, dataList, labels, "GS Energy", f"GS Energy vs. g - b={b}", outPath+f"GSEnergy_b{b}_all.jpg")
    plot_values(1, noMagnus, dataList, labels, "GS Energy", f"GS Energy vs. g - b={b}", outPath+f"GSEnergy_b{b}_nomagnus.jpg")
    plot_values(1, order_comparison, dataList, labels, "GS Energy", f"GS Energy vs. g - b={b}", outPath+f"GSEnergy_b{b}_order.jpg")

    # Plot num steps vs. g - index 2
    plot_values(2, indexList, dataList, labels, "s", f"s vs. g - b={b}", outPath+f"num_steps_b{b}_all.jpg")
    plot_values(2, noMagnus, dataList, labels, "s", f"s vs. g - b={b}", outPath+f"num_steps_b{b}_nomagnus.jpg")
    plot_values(2, order_comparison, dataList, labels, "s", f"s vs. g - b={b}", outPath+f"num_steps_b{b}_order.jpg")

    # Plot time vs. g - index 3
    plot_values(3, indexList, dataList, labels, "Time Spent (seconds)", f"Time per Flow vs. g - b={b}", outPath+f"time_b{b}_all.jpg")
    plot_values(3, noMagnus, dataList, labels, "Time Spent (seconds)", f"Time per Flow vs. g - b={b}", outPath+f"time_b{b}_nomagnus.jpg")
    plot_values(3, order_comparison, dataList, labels, "Time Spent (seconds)", f"Time per Flow vs. g - b={b}", outPath+f"time_b{b}_order.jpg")

    # Plot RAM use vs. g - index 4
    plot_values(4, indexList, dataList, labels, "RAM (kb)", f"Memory Usage vs. g - b={b}", outPath+f"RAM_b{b}_all.jpg")
    plot_values(4, noMagnus, dataList, labels, "RAM (kb)", f"Memory Usage vs. g - b={b}", outPath+f"RAM_b{b}_nomagnus.jpg")
    plot_values(4, order_comparison, dataList, labels, "RAM (kb)", f"Memory Usage vs. g - b={b}", outPath+f"RAM_b{b}_order.jpg")

#------------------------------------------------------------------------------
# make executable
#------------------------------------------------------------------------------
if __name__ == "__main__": 
  main()