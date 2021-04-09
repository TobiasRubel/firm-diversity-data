import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#df = pd.read_csv('threeDigitSICs.csv',index_col=0).T
df = pd.read_csv('yearlyFourDigitSICcount.csv', index_col=[0])



#cuts = [1,2,3,4,5,10,15,20,25,50,75,100,150,250]
cuts = [1]

counts = {}
for cut in cuts:
	for year in df.columns[4:]:
		count=0
		for row in df.index:
			if df.loc[row,year] >= cut:
				count+=1
		counts[str(year)] = count#/sum(df.iloc[row])
	out = pd.DataFrame(counts,index=[0]).T
	out.plot()
	plt.xlabel('year')
#	plt.ylabel('# SICs')
#	plt.ylim(0,0.065)
#out = pd.DataFrame(entros,index=[0]).T.iloc[3:]
#out = pd.DataFrame({'year':out.index,'#D':out.iloc[:,0]})
#out = out.set_index(pd.Series(range(0,len(out.index))))
#out[['year']] = pd.to_numeric(out['year'],downcast='integer')
#out.plot()
plt.xlabel('year')
plt.xticks(np.arange(1995,2017,step=5))
sns.regplot(x="year", y='#D', data=out.reset_index())
plt.title('SIC4'+ '  pearsonr=' + str(stats.pearsonr(range(0,len(out)),out.iloc[:,1])[0])[0:6] +',p=' + str(stats.pearsonr(range(0,len(out)),out.iloc[:,1])[1])[0:5])

plt.tight_layout()
plt.savefig(inFile + 'NonEmptyBinsX1Raw.pdf')



#sns.regplot(x="year", y='#D', data=out.reset_index())

plt.ylabel('#D')
#plt.ylim(0)
sns.set_theme(color_codes=True)
plt.title('SIC4'+ '  pearsonr=' + str(stats.pearsonr(range(0,len(out)),out.iloc[:,1])[0])[0:6] +',p=' + str(stats.pearsonr(range(0,len(out)),out.iloc[:,1])[1])[0:5])

plt.tight_layout()
plt.savefig('SIC4NonEmptyBinsX1Raw.pdf')
plt.show()


'''
	try: 
		if (str(stats.pearsonr(range(1997,2019),out[0])[1])[18] == 'e'):
			plt.title('Normalized Occurrences>=' + str(cut) + ' p=' + str(stats.pearsonr(range(1997,2019),out[0])[1])[:6]
			+ str(stats.pearsonr(range(1997,2019),out[0])[1])[18:]
			+ '  r=' + str(stats.pearsonr(range(1997,2019),out[0])[0])[:6])	
			print('e')
		else:
			plt.title('Normalized Occurrences>=' + str(cut) + ' p=' + str(stats.pearsonr(range(1997,2019),out[0])[1])[:6]
			+ '  r=' + str(stats.pearsonr(range(1997,2019),out[0])[0])[:6])
	except:	
		plt.title('Normalized Occurrences>=' + str(cut) + ' p=' + str(stats.pearsonr(range(1997,2019),out[0])[1])[:6]
		+ '  r=' + str(stats.pearsonr(range(1997,2019),out[0])[0])[:6])
	
	plt.savefig("NormalizedFrom1997SameScaleSICsWithMoreThan"+str(cut)+"occurences.png")
	plt.close()
'''

	'''plt.title('P=' 
		+ str(stats.ttest_1samp(out,26)[1][0])
		+ ' r=' +str(stats.pearsonr(out.index,out[0])[0]))

		(str(stats.pearsonr(range(1993,2019),out[0])[1])[18] == 'e')'''