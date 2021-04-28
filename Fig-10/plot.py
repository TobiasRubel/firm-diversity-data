import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
def plotbar():
    font = {'size'   : 20}
    matplotlib.rc('font', **font)
    PVDM = pd.read_csv('PVDM.csv',index_col=0)
    BOOL = pd.read_csv('bool.csv',index_col=0)
    TFIDF = pd.read_csv('tfidf.csv',index_col=0)
    #SIC4 = pd.read_csv('SIC4tree.csv',index_col=0)
    means = [np.mean(x.stack()) for x in [BOOL,PVDM,TFIDF,SIC4]]
    stdev = [np.std(x.stack()) for x in [BOOL,PVDM,TFIDF,SIC4]]
    print(means)
    print(stdev)
    fig = plt.Figure()
    ax = plt.subplot(111)
    ax.barh([1,2,3,4],means,xerr=stdev,color=["#9C3848","#47A8BD","#FF9A47","#2C6E49"],capsize=4)
    ax.plot([1 for x in np.arange(0.5,5,1.0)],[x for x in np.arange(0.5,5,1.0)],ls='--',color="#4D3F78",linewidth=3)
    ax.set_ylim(0.5,4.5)
    ax.set_yticks([1,2,3,4])
    ax.set_yticklabels(['Boolean','PV-DM','TF-IDF','SIC-4 Tree'])
    ax.set_xlabel('SIC Industry Specificity')
    ax.set_xscale('log')
    plt.tight_layout()
    plt.savefig('barh.pdf')


if __name__ == "__main__":
    plotbar()

