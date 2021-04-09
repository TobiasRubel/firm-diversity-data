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
    small = float(df.min())
    count=0
    for sic in (df.values/small):
    	if sic>(1):
    		count+=1
    entros[str(file)[11:15]] = count


out = pd.DataFrame(entros,index=[0]).T.iloc[3:]
out = pd.DataFrame({'Year':out.index,'#D':out.iloc[:,0]})
out = out.set_index(pd.Series(range(0,len(out.index))))
out[['Year']] = pd.to_numeric(out['Year'],downcast='integer')

plt.style.use('default')
plt.xlabel('Year')
#plt.gca().set_facecolor('white')
plt.xlim(1995,2020)
plt.xticks(np.arange(1995,2017,step=5))
#sns.set_theme(default, color_codes=True)
sns.regplot(x="Year", y='#D', data=out,ci=90, truncate=False)
plt.ylabel('$^{\#}$D')
plt.title('PV-DM'+ '  pearsonr=' + str(stats.pearsonr(range(0,len(out)),out.iloc[:,1])[0])[0:6] +',p=' + str(stats.pearsonr(range(0,len(out)),out.iloc[:,1])[1])[0:5])
#plt.rcParams["axes.edgecolor"] = "0.15"
#plt.rcParams["axes.linewidth"] = 1.25
#plt.grid(False)
plt.tight_layout()


plt.savefig(inFile + 'NonEmptyBinsX1Raw.pdf')
plt.show()



'''
if 'e' in str(stats.pearsonr(range(0,len(out)),out.iloc[:,1])[1]):
	plt.title('SIC4'+ '  pearsonr=' + str(stats.pearsonr(range(0,len(out)),out.iloc[:,1])[0])[0:6] +',p=0.000')
else:
	plt.title('SIC4'+ '  pearsonr=' + str(stats.pearsonr(range(0,len(out)),out.iloc[:,1])[0])[0:6] +',p=' + str(stats.pearsonr(range(0,len(out)),out.iloc[:,1])[1])[0:5])
'''