import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from scipy.stats import entropy
import os
import sys

entros = {}
inFile = r'C16'
for file in os.scandir(inFile):
    df = pd.read_csv(file, index_col=[0])
    normalizer = []
    for numnum in df.values:
        normalizer.append(sum(df.values)[0]/len(df))
    entros[str(file)[11:15]] = entropy(df)[0]/entropy(normalizer)


out = pd.DataFrame(entros,index=[0]).T.iloc[3:]
out = pd.DataFrame({'year':out.index,'$_H$D':out.iloc[:,0]})
out = out.set_index(pd.Series(range(0,len(out.index))))
out[['year']] = pd.to_numeric(out['year'],downcast='integer')

'''
df = out
font = {'size'   : 16}
matplotlib.rc('font', **font)
x = list(df['year'])
y = list(df['$_H$D'])
sns.regplot(x='index',y='$_H$D',data=df.reset_index())
corr, p = scipy.stats.pearsonr(x, y)
plt.title('{} pearsonr={:.3f},p={:.3f}'.format('TF-IDF',corr,p))
plt.xlabel('Year')
plt.ylabel('D')
plt.tight_layout()
plt.show()'''



plt.xticks(np.arange(1995,2017,step=5))

g = sns.regplot(x="year", y='$_H$D', data=out.reset_index())
plt.xlabel('year')
#g.set(xlim=(1995,2020))
plt.ylabel('$_H$D')
g.margins(x=0.5)


#plt.ylim(0)
sns.set_theme(color_codes=True)



plt.title('PV-DM'+ '  pearsonr=' + str(stats.pearsonr(range(0,len(out)),out.iloc[:,1])[0])[0:6] +',p=' + str(stats.pearsonr(range(0,len(out)),out.iloc[:,1])[1])[0:5])
'''plt.savefig(inFile + 'EntropyAbundVecRaw.png')'''
plt.tight_layout()
#plt.savefig(inFile + 'EntropyFinal1997Raw1.pdf')
plt.show()

