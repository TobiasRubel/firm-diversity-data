import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv('yearlyFourDigitSICcount.csv',index_col=0)
thresh = 1

'''
for no differences between years (eg. all SICs that appear at all)
change thresh to 0.
Plus, change the file name and add a break and whatnot
'''

def digiter(si1, si2):
	count = 0
	print(str(si1))
	if si1 == si2 and si1 == 0:
		count = len(df.index)
	elif len(str(si1)) == 3 or len(str(si2)) == 3:
		if(str(si1)[0] == str(si2)[0]):
			if(str(si1)[1] == str(si2)[1]):
				if((str(si1)[2] == str(si2)[2])):
					for SIC in df.index:
						try:
							if(str(SIC)[1]==str(si1)[1] and str(SIC)[0] == str(si1)[0]) and str(SIC)[2] == str(si1)[2]:
								count+=1
						except:pass
				for SIC in df.index:
						try:
							if(str(SIC)[1]==str(si1)[1] and str(SIC)[0] == str(si1)[0]):
								count+=1
						except:pass
			else:
				for SIC in df.index:
						try:
							if(str(SIC)[0] == str(si1)[0]):
								count+=1
						except:pass
		else:
			count = len(df.index)	
	elif(str(si1)[0] == str(si2)[0]):
		try:
			if(str(si1)[1] == str(si2)[1]):
				if(str(si1)[2] == str(si2)[2]):
					try:
						if(str(si1)[3] == str(si2)[3]):
							count = 1
						else:
							for SIC in df.index:
								try:
									if(str(SIC)[1]==str(si1)[1] and str(SIC)[0] == str(si1)[0] and str(SIC)[2] == str(si1)[2]):
										count+=1
								except:pass
					except:
						for SIC in df.index:
							try:
								if(str(SIC)[1]==str(si1)[1] and str(SIC)[0] == str(si1)[0]):
									count+=1
							except:pass 
				else: 
					for SIC in df.index:
						try:
							if(str(SIC)[1]==str(si1)[1] and str(SIC)[0] == str(si1)[0]):
								count+=1
						except:pass
							
			else:
				for SIC in df.index:
					try:
						if(str(SIC)[0] == str(si1)[0]):
							count+=1
					except:pass	
		except:
			for SIC in df.index:
					try:
						if(str(SIC)[0] == str(si1)[0]):
							count+=1
					except:pass	
	else: count = len(df.index)
	return 1/count


out = {}
for year in df.columns:
	out = {}
	for sic1 in df.index:
		subout = {}
		if df.loc[sic1][year]>=thresh and len(str(sic1))>=4:
			for sic2 in df.index:
				if df.loc[sic2][year]>=thresh and len(str(sic2))>=4:
					subout[sic2] = digiter(sic1,sic2)
			out[sic1] = subout
	pd.DataFrame(out).to_csv(str(year) + 'SimpleModelSIC4DistanceVecs.csv')
	#pd.DataFrame(out).to_csv('overallSimpleModelSIC4DistanceVecs.csv')
