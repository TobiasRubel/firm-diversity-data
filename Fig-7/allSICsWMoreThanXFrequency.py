import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from scipy.stats import entropy
import math

#df = pd.read_csv('threeDigitSICs.csv', index_col=[0])
df = pd.read_csv('yearlyFourDigitSICcount.csv', index_col=[0]).T

#cuts = [1,2,3,4,5,10,15,20,25,50,75,100,150,250]
cuts=[1]


dicut = {}
dicut1={}
x = 0
for cut in cuts:
	dicut[cut] = {}
	dicut1[cut]={}
	for row in range(0,len(df.index)):
		count=0
		for word in df.columns:
		#for word in df.columns[:]:
			if df.iloc[row][word] >= cut:
				count+=1
		dicut1[cut][str(row+1993)] = count#/sum(df.iloc[row])
		dicut[cut][str(row+1993)] = entropy(df.iloc[row])/math.log(count)



out = pd.DataFrame(dicut).iloc[4:25]
#out = pd.DataFrame(dicut1).iloc[4:25]#/pd.DataFrame(dicut).iloc[4:25]
out = pd.DataFrame({'Year':out.index,'#_D':out.iloc[:,0]})
out = out.set_index(pd.Series(range(0,len(out.index))))
out[['Year']] = pd.to_numeric(out['Year'],downcast='integer')
#out.plot()
plt.style.use('default')
plt.xlabel('Year')
plt.xticks(np.arange(1995,2017,step=5))
plt.xlim(1995,2020)
#sns.regplot(x="Year", y='H_D', data=out.reset_index(),ci = 90,truncate=False)
sns.regplot(x="Year", y='#_D', data=out.reset_index(),ci = 90,truncate=False)
#plt.xlim(1995,2020)
#sns.regplot(x="year", y='#D', data=out.reset_index())
plt.ylabel('$^{\#}$D')
#plt.ylabel('$^H$D')
#plt.ylim(0)
sns.set_theme(color_codes=True)



if stats.pearsonr(range(0,len(out)),out.iloc[:,1])[1] < 0.001:
	plt.title('SIC4'+ '  pearsonr=' + str(stats.pearsonr(range(0,len(out)),out.iloc[:,1])[0])[0:6] +',p<0.001')
else:
	plt.title('SIC4'+ '  pearsonr=' + str(stats.pearsonr(range(0,len(out)),out.iloc[:,1])[0])[0:6] +',p=' + str(stats.pearsonr(range(0,len(out)),out.iloc[:,1])[1])[0:5])


plt.tight_layout()
plt.savefig('4digSicNormalizedCountsEntropySIC4to2017.pdf')
plt.show()