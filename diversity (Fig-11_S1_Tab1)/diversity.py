import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from scipy.stats.stats import pearsonr

#Path to similarity matrices
#PATH_TO_C='/home/annambiar/10k2v/Hybrid/boolean/nouns_10k_boolean_0.2-100_class_sims_' #replace this with the appropriate directory
#PATH_TO_C='/home/annambiar/10k2v/Hybrid/PVDM/10k2v_sim_'
PATH_TO_C='/home/annambiar/10k2v/Hybrid/TFIDF/nouns_10k_tfidf_1.0-100_class_sims_'

#Path to abundance vectors
PATH_TO_SIC = '/home/annambiar/10k2v/C-SIC/sic-abundance-'

Q=0
ADJUSTED = False

#Save outputs
FIG_NAME = 'tfidf_diversity_q=0.pdf'
CSV_NAME = 'tfidf_diversity_q=0.csv'

def diversity(q,sims,abund):
    if q!=1:
        out=0
        for i in abund.index:
            out+= abund['abundance'][i]*((np.dot(sims[str(i)].values,abund['abundance']))**(q-1))
        out = out**(1/(1-q))
    else:
        denom = 1
        for i in abund.index:
            denom = denom * (np.dot(sims[str(i)].values,abund['abundance'])**(abund['abundance'][i]))
        out = 1/denom
    return out

if __name__ == "__main__":
    fig= plt.figure()
    ax = plt.axes()
    out = []
    for i in range(21):
        year = str(1997+i)
        abundances = pd.read_csv(PATH_TO_SIC+year+'.csv',index_col=0)
        sims = pd.read_csv(PATH_TO_C+year+'.csv',index_col=0)
        
        classes = [str(i) for i in range(len(sims))]
        sims.columns = classes
        sims.index=classes
        a_series = (abundances != 0).any(axis=1)
        abundances =abundances.loc[a_series]
        classes = [str(i) for i in abundances.index]
        sims = sims[classes]
        sims = sims.loc[classes]

        if ADJUSTED:
            div = diversity(Q,sims,abundances)/len(classes)
        else:
            div = diversity(Q,sims,abundances)
        out.append([1997+i,div])

    div_dfs = pd.DataFrame(out,columns=['year','D'])
    m, b = np.polyfit(div_dfs['year'],div_dfs['D'], 1)
    sns.regplot(div_dfs['year'],div_dfs['D'], ci=90)
    #ax.set_ylabel('D',fontsize=18)
    #ax.set_xlabel('year',fontsize=18)
    plt.rcParams.update({'font.size': 22})
    x_ticks = np.arange(1995, 2016, 5)
    plt.xlim(left=1994.5)
    plt.xticks(x_ticks)
    plt.tight_layout()
    plt.savefig(FIG_NAME)
    div_dfs.to_csv(CSV_NAME)
    print(pearsonr(div_dfs['year'],div_dfs['D']))