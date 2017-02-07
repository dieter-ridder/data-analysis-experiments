# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 21:15:58 2016

@author: Charlotte
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from os import listdir
from os.path import isfile, join



'''
Names are delivered Bezirk by Bezirk
so we have to read them one by one
'''
all_2015_df = pd.DataFrame({
        'anzahl': [],
        'vorname': [],
        'geschlecht': [],
        'bezirk': []
    })


path = 'D:/udacity_Intro_DataAnalysis/Vornamen/'

onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

def read_bezirk(filename):
    bezirk_df = pd.read_csv(path+filename,sep=';')
    bezirk_df.dropna(how='any',inplace =True)

    (name,unused)=filename.split('.')
    bezirk_df['bezirk']=name
    return bezirk_df
def create_text_gender (abbr):
    if abbr == 'w':
        return 'girls'
    return 'boys'

def percentage (param):
    return param/param.sum()
    
    
    
for f in onlyfiles:
    next_bezirk=read_bezirk(f)
    all_2015_df=all_2015_df.append(next_bezirk,ignore_index=True)

#seperate according to gender
all_2015_w_df=all_2015_df[all_2015_df['geschlecht']=='w']
all_2015_m_df=all_2015_df[all_2015_df['geschlecht']=='m']

print('Overview: \nBabies in Berlin 2015: sum: {}, girls: {}, boys: {}'.format(\
      all_2015_df['anzahl'].sum(),all_2015_w_df['anzahl'].sum(),all_2015_m_df['anzahl'].sum()))

plt.figure(0)
plt.pie([all_2015_m_df['anzahl'].sum(),all_2015_w_df['anzahl'].sum() ],labels=['boys', 'girls'],autopct='%.0f%%')
plt.show()



# babies by bezirk
print ('Babies by Bezirk')
group_by_bezirk=all_2015_df.groupby(by=['bezirk','geschlecht'],as_index=False).sum()

pivot_by_bezirk=group_by_bezirk.pivot(index='bezirk', columns='geschlecht', values='anzahl'  )
pivot_by_bezirk['sum']=pivot_by_bezirk['m']+pivot_by_bezirk['w']
pivot_by_bezirk['perc_m']=pivot_by_bezirk['m']/pivot_by_bezirk['sum']
pivot_by_bezirk['perc_m'] = pivot_by_bezirk['perc_m'].map('{:,.1%}'.format)

pivot_by_bezirk['perc_w']=pivot_by_bezirk['w']/pivot_by_bezirk['sum']
pivot_by_bezirk['perc_w'] = pivot_by_bezirk['perc_w'].map('{:,.1%}'.format)
pivot_by_bezirk['more_girls']=pivot_by_bezirk['w']>pivot_by_bezirk['m']

pivot_by_bezirk.sort_values(by ='sum',ascending=False,inplace=True)
print(pivot_by_bezirk)

plt.figure(1)
plt.pie(pivot_by_bezirk['sum'],labels=pivot_by_bezirk.index,autopct='%.0f%%')
plt.show()

pivot_by_bezirk['perc_m']=pivot_by_bezirk['m']/pivot_by_bezirk['sum']
pivot_by_bezirk['perc_w']=pivot_by_bezirk['w']/pivot_by_bezirk['sum']
plt.figure(2)
ind = np.arange(len(pivot_by_bezirk))    # the x locations for the groups
width = 0.4      # the width of the bars: can also be len(x) sequence
plt.bar(ind-0.2,pivot_by_bezirk['perc_m'],width=0.4,color='b', label='Boys')
plt.bar(ind+0.2,pivot_by_bezirk['perc_w'],width=0.4,color='g',label='Girls')
plt.xticks(ind + width/2., pivot_by_bezirk.index,rotation=90)
plt.ylabel('percentage')
plt.title('gender: percentage per bezirk')
plt.legend()
plt.show()

# female Names
pivot_by_name_w=all_2015_w_df.pivot(index='vorname', columns='bezirk', values='anzahl'  )
pivot_by_name_w.fillna(0.0,inplace=True)
pivot_by_name_w['Berlin']=pivot_by_name_w.sum(axis=1)
pivot_by_name_w=pivot_by_name_w.apply(percentage)
pivot_by_name_w.sort_values(by='Berlin',ascending=False, inplace=True)
pivot_by_name_w['cumulated']=pivot_by_name_w['Berlin'].cumsum()


# male Names
pivot_by_name_m=all_2015_m_df.pivot(index='vorname', columns='bezirk', values='anzahl'  )
pivot_by_name_m.fillna(0.0,inplace=True)
pivot_by_name_m['Berlin']=pivot_by_name_m.sum(axis=1)
pivot_by_name_m=pivot_by_name_m.apply(percentage)
pivot_by_name_m.sort_values(by='Berlin',ascending=False, inplace=True)
pivot_by_name_m['cumulated']=pivot_by_name_m['Berlin'].cumsum()

print ('NAMES\ndifferent names used: {}, for girls: {}, for boys: {}'.format(\
       len(pivot_by_name_m)+len(pivot_by_name_w), len(pivot_by_name_w), len(pivot_by_name_m)))

values=np.array([0, 0, 0, 0, 0])
no_names = pd.DataFrame({
                          'girls': values,
                          'boys': values},
                  index=[0.1,0.2,0.3,0.5,0.75])

def find_limits(limits, series):
    result=[]
    l=0
    i=0
    last_s=0.0
    for s in series:
        if s > limits[l]:
            result.append(i)
            l+=1
            if l >= len(limits):
                break
        last_s=s
        i+=1
        
    return result
    
no_names['girls']=find_limits(no_names.index, pivot_by_name_w['cumulated'])
no_names['boys']=find_limits(no_names.index, pivot_by_name_m['cumulated'])
print '\nHow many names are necessary to name x percent?\n',no_names

plt.figure(3)
pd.options.display.float_format = '{:,.2f}'.format

plt.plot(np.arange(len(pivot_by_name_w)),pivot_by_name_w['cumulated'],color='g', label='girls')
plt.plot(np.arange(len(pivot_by_name_m)),pivot_by_name_m['cumulated'],color='b', label='boys')
plt.ylabel('percentage')
plt.xlabel('number of names')
plt.xscale('log')
for i in no_names.index:
    xGirls=no_names['girls'].loc[i]
    xBoys=no_names['boys'].loc[i]
    plt.axhline(y=i,linewidth=1, color='gray', linestyle='--')
    plt.fill_between([xGirls,xBoys],[1.0,1.0], color='gray', alpha=0.5)


plt.legend(loc='lower right')

plt.title('how many names are necessary to name 10%/30%/50% of the year?')
plt.show()





#top ten names
top_ten_w=pivot_by_name_w.iloc[:6]
top_ten_w=top_ten_w.rank(method='max')
top_ten_w=top_ten_w.drop('cumulated',axis=1)
#print(top_ten_w.head())
top_ten_w=top_ten_w.T
print(top_ten_w.head(20))

#top_ten_w.plot()

ind = np.arange(len(top_ten_w))    # the x locations for the groups
color=['b','g','r','y','b','m']



i=0
for c in top_ten_w.columns:
    plt.figure(4)
    plt.bar(ind+i*0.1,top_ten_w[c],width=0.1,label=c,color=color[i], alpha=1.0-i*0.15)
    
#    plt.figure(5)
#    plt.step(ind,top_ten_w[c],label=c,color=color[i])

#    plt.figure(6)
#    plt.scatter(ind,top_ten_w[c],label=c,color=color[i])

#    plt.figure(7)
#    plt.plot(ind,top_ten_w[c],label=c,color=color[i], \
#             marker='H',linestyle='--')
             #drawstyle='steps-mid',\

    plt.figure(8)
    plt.vlines(ind+i*0.1,ymin=0,ymax=top_ten_w[c],label=c,color=color[i],linewidth=2, alpha=1.0-i*0.15)

    i+=1
##plt.scatter(ind,top_ten_w['Marie'])
plt.figure(4)
plt.legend(bbox_to_anchor=(1.2, 1.0))
plt.xticks(ind+0.3,top_ten_w.index,rotation=90)
#plt.figure(5)
#plt.legend()
#plt.figure(6)
#plt.legend()
plt.figure(8)
plt.legend(bbox_to_anchor=(1.2, 1.0))
plt.xticks(ind+0.3,top_ten_w.index,rotation=90)
#plt.xticks(list(top_ten_w.index),rotation=90)
