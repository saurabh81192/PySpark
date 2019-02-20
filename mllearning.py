'''
Created on Feb 11, 2019

@author: saurabh.chakraborty
'''
from pyspark import SQLContext, SparkConf, SparkContext
import matplotlib.pyplot as plt
import numpy as np
from numpy import polyfit


#
conf = SparkConf().setMaster('local').setAppName('ML_learning')
sc = SparkContext(conf=conf)
sqlcontext = SQLContext(sc)
 
data = sqlcontext.read.csv(path='salary_d.csv', header = True, inferSchema = True)
data.show()

x1 = data.toPandas()['Expr_yrs'].values.tolist()
y1 = data.toPandas()['Salary'].values.tolist()

# x1 = data['YearsExperience'].values
# y1 = data['Salary'].values

plt.scatter(x1, y1, color='red', s=30)

plt.xlabel('Expr_yrs')
plt.ylabel('Salary')
plt.title('Linear Regression')

p1 = polyfit(x1, y1, 1)
print(p1)
plt.plot(x1, np.polyval(p1,x1), 'g-' )
# plt.plot(np.unique(x1), np.poly1d(np.polyfit(x1, y1, 1))(np.unique(x1)))
plt.show()

