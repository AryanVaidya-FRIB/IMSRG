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
def plot_flow(j, indexList, dataList, colorList, labels, metric, title, saveName):
    plt.figure()
    for i in reversed(indexList):
        plt.plot(dataList[i].iloc[:,0].to_numpy(), dataList[i].iloc[:,j].to_numpy(), label = labels[i], color = colorList[i])
    
    plt.legend()
    plt.xlabel("s")
    plt.ylabel(metric)
    plt.title(title)
    plt.savefig(saveName)
    return

def plot_difference(indexList, dataList, colorList, labels, metric, title, saveName):
    plt.figure()
    final_E = dataList[0].iloc[-1,1]
    for i in reversed(indexList):
        if i == 0:
            continue
        plt.plot(dataList[i].iloc[:,0].to_numpy(), dataList[i].iloc[:,1].to_numpy()-final_E, label = labels[i], color = colorList[i])
    
    plt.legend()
    plt.xlabel("s")
    plt.ylabel(metric)
    plt.title(title)
    plt.savefig(saveName)
    return

#------------------------------------------------------------------------------
# Main Program
#------------------------------------------------------------------------------

def main():
    delta = "1.0"
    g     = "0.5"
    b     = "0.5"
    podS  = "0.5"

    models = ["Galerkin", "OpInf"]

    # Files to pull
    dPath      = "/mnt/c/Users/aryan/Documents/MSU_FRIB/IMSRG/proper_orth_decomp/"
    filePath   = dPath + f"model_comparison_s{podS}/"
    outPath    = dPath + "plots/"
    fileString_g = f"imsrg-Galerkin_d{delta}_g{g}_b{b}_N4_"
    fileString = f"imsrg-OpInf_d{delta}_g{g}_b{b}_N4_"
    
    # List of each benchmark
    flows = ["pod_rank15","pod_rank2", "pod_rank6", "pod_rank20", "pod_rank51"]
    labels = ["Galerkin 15", "OpInf 2", "OpInf 6", "OpInf 20", "OpInf 51"]
    colorList = ['C0', 'C2', 'C3', 'C4', 'C5']
    indexList = range(len(flows))

    # Load datafiles as a list of Pandas Dataframes
    dataList = []
    dataList.append(pd.read_csv(filePath+fileString_g+flows[0]+"_fullflow.csv", index_col=0))
    for i in indexList:
        if i+1 == len(indexList):
            continue
        dataList.append(pd.read_csv(filePath+fileString+flows[i+1]+"_fullflow.csv", index_col=0))

    # Plot energy vs. s
    plot_flow(1, indexList, dataList, colorList, labels, "GS Energy", f"GS Energy vs. Time - b={b}, POD s={podS}", outPath+f"GSEnergy_b{b}_s{podS}_SVDPOD_OpInf.jpg")

    # Plot GammaOD vs. s
    plot_flow(2, indexList, dataList, colorList, labels, "Gamma OD", f"GammaOD vs. Time - b={b}, POD s={podS}", outPath+f"GammaOD_b{b}_s{podS}_SVDPOD_OpInf.jpg")

    # Plot difference in original vs. final energy for each rank
    plot_difference(indexList, dataList, colorList, labels, "Energy Difference", f"Energy Difference vs. Time - b={b}", outPath+f"Ediff_b{b}_s{podS}_SVDPOD_OpInf.jpg")

#------------------------------------------------------------------------------
# make executable
#------------------------------------------------------------------------------
if __name__ == "__main__": 
  main()