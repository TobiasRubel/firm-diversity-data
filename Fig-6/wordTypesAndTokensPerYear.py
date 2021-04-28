import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tokens = pd.read_csv('yearlyWordCount.csv')
types = pd.read_csv('kindaTypes.csv.txt',index_col=[0])
#types1 = pd.read_csv('yearlyWordTypes.csv').T
docs = pd.read_csv('yearlyDocuments.csv')




tokcount = {}
typcount = {}
for row in range(4,25):
	tokcount[str(row+1993)] = tokens.iloc[row,1:].sum()/docs.iloc[row,1]
	typcount[str(row+1993)] = types.iloc[row,0]/docs.iloc[row,1]

df = pd.DataFrame({'Tokens/Documents':pd.Series(tokcount), 'Types/Documents':pd.Series(typcount)})

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Year')
ax1.set_ylabel('Tokens/Documents', color=color)
ax1.scatter(df.index,df['Tokens/Documents'], color=color)
#ax1.tick_params(axis='x', labelrotation=45)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Average Types/Document', color=color)  # we already handled the x-label with ax1
ax2.scatter(df.index,df['Types/Documents'], color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.xaxis.set_major_locator(plt.MaxNLocator(5))
#ax2.xaxis.set_major_locator(ax2.xaxis,[1995,2000,2005,2010,2015])
plt.title('Word Types and Word Tokens')
plt.tight_layout()

plt.savefig('wordTypesAndTokensPerYearAdjusted.pdf')
plt.show()

