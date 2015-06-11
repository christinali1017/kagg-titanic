#This script looks at the survival rate for those with the most common names

import numpy as np
import pandas as pd

# from kaggleaux import predict as ka # see github.com/agconti/kaggleaux for more details
df = pd.read_csv("train.csv") 

#drop these two columns
df = df.drop(['Ticket','Cabin'], axis=1) 

# Remove NaN values
df = df.dropna() 

#Extract the first name from passenger name
df['FirstName'] = df['Name'].str.extract('(Mr\. |Miss\. |Master. |Mrs\.[A-Za-z ]*\()([A-Za-z]*)')[1]

#pull out the passengers that have popular names (> 10 occurances)
popularnames = df[df['FirstName'].isin(df['FirstName'].value_counts()[df['FirstName'].value_counts() > 10].index)]


test = popularnames.groupby(['Sex', 'FirstName'])
print(test['Survived'].agg([np.mean, len]))




#calculate the surival rate by popular name
#ax = (dfPassengersWithPopularNames.groupby('FirstName').Survived.sum()/dfPassengersWithPopularNames.groupby('FirstName').Survived.count()).order(ascending=False).plot(kind='barh',fontsize=15)

#set y axis label and save to png for display below
#fig = ax.get_figure()
#fig.savefig('figure.png')

print('Finished printing ... ')
