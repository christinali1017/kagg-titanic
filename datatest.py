# -*- coding: utf-8 -*-   # for ascII encoding
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric import smoothers_lowess
from pandas import Series, DataFrame
from patsy import dmatrices
from sklearn import datasets, svm


"""
 kind of plot
'bar’ or ‘barh’ for bar plots
‘hist’ for histogram
‘box’ for boxplot
‘kde’ or 'density' for density plots
‘area’ for area plots
‘scatter’ for scatter plots
‘hexbin’ for hexagonal bin plots
‘pie’ for pie plots

"""

# from kaggleaux import predict as ka # see github.com/agconti/kaggleaux for more details
df = pd.read_csv("train.csv") 

#drop these two columns
df = df.drop(['Ticket','Cabin'], axis=1) 

# Remove NaN values
df = df.dropna() 

 
alpha=alpha_scatterplot = 0.2 
alpha_bar_chart = 0.55

### survival
df.Survived.value_counts().plot(kind='bar', alpha=alpha_bar_chart)
plt.ylabel("Number of People")
plt.title("Overall distribution of survival, (1 = Survived)")  


### Pclass

df.Pclass.value_counts().plot(kind="barh", alpha=alpha_bar_chart)
plt.ylabel("Class")
plt.title("Class Distribution")


###Survival by Pclass
ax1 = plt.subplot2grid((2,2),(0,0))
df.Survived[df.Pclass == 3].value_counts().plot(kind='barh', color='#01DFA5',label='Class 3')
plt.title("survival of class 3, (1 = Survived)"); 
plt.subplot2grid((2,2),(0,1))
df.Survived[df.Pclass == 2].value_counts().plot(kind='barh', color='#FA2379',label='Class 2')
plt.title("survival of class 2, (1 = Survived)"); 

plt.subplot2grid((2,2),(1, 0))
df.Survived[df.Pclass == 1].value_counts().plot(kind='barh',label='Class 1')
plt.title("survival of class 1, (1 = Survived)"); 

plt.subplot2grid((2,2),(1, 1))
df.Survived[df.Pclass == 3].value_counts().plot(kind='barh', color='#01DFA5',label='Class 3')
df.Survived[df.Pclass == 2].value_counts().plot(kind='barh', color='#FA2379',label='Class 2')
df.Survived[df.Pclass == 1].value_counts().plot(kind='barh',label='Class 1')
plt.title("survival by Pclass, (1 = Survived)"); plt.legend(loc='best')


###Age
df['Age'].plot(kind="kde")  # density
plt.xlabel("Age")
plt.title("Age distribution");


###Age distribution by class

df.Age[df.Pclass == 1].plot(kind='kde')    
df.Age[df.Pclass == 2].plot(kind='kde')
df.Age[df.Pclass == 3].plot(kind='kde')
plt.xlabel("Age")    
plt.title("Age Distribution by classes")
plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best') 



###Survive by age
df.Age[df.Survived == 1][df.Pclass==1].plot(kind='kde')   
plt.xlabel("Age")    
plt.title("Survive by age")


# Survive by age and class
df.Age[df.Survived == 1][df.Pclass==1].plot(kind='kde')   
df.Age[df.Survived == 1][df.Pclass == 2].plot(kind='kde')
df.Age[df.Survived == 1][df.Pclass == 3].plot(kind='kde') 
plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best')
plt.xlabel("Age")
plt.title("Survive distribution by age and Pclass");


###Age all

plt.figure(figsize=(12,6))
ax1 = plt.subplot2grid((2,2),(0,0))
###Age
df['Age'].plot(kind="kde")  # density
plt.xlabel("Age")
plt.title("Age distribution");
###Age distribution by class
plt.subplot2grid((2,2),(0,1))
df.Age[df.Pclass == 1].plot(kind='kde')    
df.Age[df.Pclass == 2].plot(kind='kde')
df.Age[df.Pclass == 3].plot(kind='kde')
plt.xlabel("Age")    
plt.title("Age Distribution by classes")
plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best') 
###Survive by age
plt.subplot2grid((2,2),(1,0))
df.Age[df.Survived == 1][df.Pclass==1].plot(kind='kde')   
plt.xlabel("Age")    
plt.title("Survive by age")
# Survive by age and class
plt.subplot2grid((2,2),(1,1))
df.Age[df.Survived == 1][df.Pclass==1].plot(kind='kde')   
df.Age[df.Survived == 1][df.Pclass == 2].plot(kind='kde')
df.Age[df.Survived == 1][df.Pclass == 3].plot(kind='kde') 
plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best')
plt.xlabel("Age")
plt.title("Survive distribution by age and Pclass");


###Survive by sex
plt.figure(figsize=(12,6))

ax1 = plt.subplot2grid((2,2),(0,0))
df.Survived.value_counts().plot(kind='barh')
plt.title("Overall distribution of survival, (1 = Survived)") 

plt.subplot2grid((2,2),(0,1))
df.Survived[df.Sex == 'male'].value_counts().plot(kind='barh',label='male')
plt.title("survival of male, (1 = Survived)"); 

plt.subplot2grid((2,2),(1, 0))
df.Survived[df.Sex == 'female'].value_counts().plot(kind='barh',label='female')
plt.title("survival of female, (1 = Survived)"); 

plt.subplot2grid((2,2),(1, 1))

df.Survived[df.Sex == 'male'].value_counts().plot(kind='barh', color='#FA2379',label='male')
df.Survived[df.Sex == 'female'].value_counts().plot(kind='barh', color='#01DFA5',label='female')
plt.title("survival by Sex, (1 = Survived)"); plt.legend(loc='best')



###class and sex
ax1 = plt.subplot2grid((2,2),(0,0))

female_highclass = df.Survived[df.Sex == 'female'][df.Pclass != 3].value_counts()
female_highclass.plot(kind='bar', label='female highclass')
ax1.set_xticklabels(["1", "0"], rotation=0)
ax1.set_xlim(-1, len(female_highclass))
plt.title("Survive with respect to Gender and Class"); plt.legend(loc='best')

ax2 = plt.subplot2grid((2,2),(0,1))

female_lowclass = df.Survived[df.Sex == 'female'][df.Pclass == 3].value_counts()
female_lowclass.plot(kind='bar', label='female, low class', color='#01DFA5')
ax2.set_xticklabels(["0","1"], rotation=0)
ax2.set_xlim(-1, len(female_lowclass))
plt.legend(loc='best')

ax3 = plt.subplot2grid((2,2),(1, 0))

male_lowclass = df.Survived[df.Sex == 'male'][df.Pclass == 3].value_counts()
male_lowclass.plot(kind='bar', label='male, low class',color='#FA2379')
ax3.set_xticklabels(["0","1"], rotation=0)
ax3.set_xlim(-1, len(male_lowclass))
plt.legend(loc='best')

ax4 = plt.subplot2grid((2,2),(1, 1))

male_highclass = df.Survived[df.Sex == 'male'][df.Pclass != 3].value_counts()
male_highclass.plot(kind='bar', label='male highclass', color='#F5D0A9')
ax4.set_xticklabels(["0","1"], rotation=0)
ax4.set_xlim(-1, len(male_highclass))
plt.legend(loc='best')



###Board all

plt.figure(figsize=(12,6))
ax1 = plt.subplot2grid((2,2),(0,0))
#
df.Embarked.value_counts().plot(kind='bar')
plt.xlabel("Embarked")
plt.title("Embarked distribution");


plt.subplot2grid((2,2),(0,1))
df.Survived[df.Embarked == 'S'].value_counts().plot(kind='bar',label='male')
plt.title("survival of board S, (1 = Survived)"); 

plt.subplot2grid((2,2),(1, 0))
df.Survived[df.Embarked == 'C'].value_counts().plot(kind='bar',label='female')
plt.title("survival of board C, (1 = Survived)"); 

plt.subplot2grid((2,2),(1, 1))
df.Survived[df.Embarked == 'Q'].value_counts().plot(kind='bar',label='female')
plt.title("survival of board Q, (1 = Survived)"); 


###class and sex
ax1=fig.add_subplot(141)
female_highclass = df.Survived[df.Sex == 'female'][df.Pclass != 3].value_counts()
female_highclass.plot(kind='bar', label='female highclass', color='#FA2479', alpha=alpha_level)
ax1.set_xticklabels(["Survived", "Died"], rotation=0)
ax1.set_xlim(-1, len(female_highclass))
plt.title("Who Survived? with respect to Gender and Class"); plt.legend(loc='best')

ax2=fig.add_subplot(142, sharey=ax1)
female_lowclass = df.Survived[df.Sex == 'female'][df.Pclass == 3].value_counts()
female_lowclass.plot(kind='bar', label='female, low class', color='pink', alpha=alpha_level)
ax2.set_xticklabels(["Died","Survived"], rotation=0)
ax2.set_xlim(-1, len(female_lowclass))
plt.legend(loc='best')

ax3=fig.add_subplot(143, sharey=ax1)
male_lowclass = df.Survived[df.Sex == 'male'][df.Pclass == 3].value_counts()
male_lowclass.plot(kind='bar', label='male, low class',color='lightblue', alpha=alpha_level)
ax3.set_xticklabels(["Died","Survived"], rotation=0)
ax3.set_xlim(-1, len(male_lowclass))
plt.legend(loc='best')

ax4=fig.add_subplot(144, sharey=ax1)
male_highclass = df.Survived[df.Sex == 'male'][df.Pclass != 3].value_counts()
male_highclass.plot(kind='bar', label='male highclass', alpha=alpha_level, color='steelblue')
ax4.set_xticklabels(["Died","Survived"], rotation=0)
ax4.set_xlim(-1, len(male_highclass))
plt.legend(loc='best')


###Show overall data
train = pd.read_csv("train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("test.csv", dtype={"Age": np.float64}, )
print("\n\nSummary statistics of training data")
print(train.describe())


###Random forest test 1
# -*- coding: utf-8 -*-   # for ascII encoding
"""
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric import smoothers_lowess
from pandas import Series, DataFrame
from patsy import dmatrices
from sklearn import datasets, svm
# import the machine learning library that holds the randomforest
import sklearn.ensemble as ske
"""

df = pd.read_csv("train.csv") 

#drop these two columns
df = df.drop(['Ticket','Cabin'], axis=1) 

# Remove NaN values
df = df.dropna() 

formula_ml = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp + Parch + C(Embarked)'

# Create the random forest model and fit the model to our training data
y, x = dmatrices(formula_ml, data=df, return_type='dataframe')
# RandomForestClassifier expects a 1 demensional NumPy array, so we convert
y = np.asarray(y).ravel()
#instantiate and fit our model
results_rf = ske.RandomForestClassifier(n_estimators=100).fit(x, y)

# Score the results
score = results_rf.score(x, y)


plt.show()