import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import entropy
import seaborn.apionly as sns


#a file with categories as rows and years as columns

#df = pd.read_csv('yearlySICcounts.csv')
df = pd.read_csv('yearlyFourDigitSICcount.csv', index_col=[0])
#df = pd.read_csv('threeDigitSICs.csv',index_col=0).T

names = ['Agriculture, etc.','Mining','Construction','Manufacturing','Utilities','Wholesale Trade','Retail Trade','Services','Nonclassifiable']

outs = {}
for year in df.columns[4:]:
	outs[year] = {}
	for name in names:
		outs[year][name] = 0
	for SIC in df.index[1:]:
		if len(str(SIC)) == 3:
			outs[year]['Agriculture, etc.'] += df.loc[SIC][year]
		elif (SIC>=1000 and SIC<=1499):
			outs[year]['Mining'] += df.loc[SIC][year]
		elif (SIC>=1500 and SIC<=1999):
			outs[year]['Construction'] += df.loc[SIC][year]
		elif SIC>=2000 and SIC<=3999:
			outs[year]['Manufacturing'] += df.loc[SIC][year]
		elif SIC>=4000 and SIC<=4999:
			outs[year]['Utilities'] += df.loc[SIC][year]
		elif SIC>=5000 and SIC<=5199:
			outs[year]['Wholesale Trade'] += df.loc[SIC][year]
		elif SIC>=5200 and SIC<=5999:
			outs[year]['Retail Trade'] += df.loc[SIC][year]
		elif SIC>=7000 and SIC<=8999:
			outs[year]['Services'] += df.loc[SIC][year]
		elif SIC>=9900 and SIC<=9999:
			outs[year]['Nonclassifiable'] += df.loc[SIC][year]
		else:
			print('something is not quite right')



plt.style.use('default')

out = pd.DataFrame(outs)
#the solution to your problems:
fig = plt.figure(num=1, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_axes([0.11, 0.15, 0.75, 0.75])
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.65, box.height])


pits = {0: plt.bar(out.columns, out.iloc[0,:])}
bot = out.iloc[0,:]
for x in [1,2,3,4,5,7,8]:
	pits[x] = plt.bar(out.columns, out.iloc[x,:],bottom=bot)
	bot+=out.iloc[x,:]


plt.xticks(np.arange(0,21,step=5), np.arange(1997,2018,step=5))
plt.legend((pits[0],pits[1],pits[2],pits[3],pits[4],pits[5],pits[7],pits[8]),
	('Agriculture, etc.','Mining','Construction','Manufacturing','Utilities','Wholesale Trade','Retail Trade','Services','Nonclassifiable'), bbox_to_anchor=(1,1),loc='upper left')

plt.xlabel('Year')
plt.ylabel('# CIKs')

plt.savefig("namedSIC4yearlyTokens.pdf")
plt.show()