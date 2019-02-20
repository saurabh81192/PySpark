'''
Created on Feb 19, 2019

@author: saurabh.chakraborty
'''
from pyspark import SQLContext, SparkConf, SparkContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

conf = SparkConf().setMaster('local').setAppName('ML_learning')
sc = SparkContext(conf=conf)
sqlcontext = SQLContext(sc)
 
data = sqlcontext.read.csv(path='bike_sharing_daily.csv', header = True, inferSchema = True)
# data.printSchema()

# Here used infer schema so printschema giving correct result. or else we could have explicitly type casted
# data2 = data.select(data.season.astype('int'))
# we will remove the unwanted columns

data2 = data.select(data.season, data.yr,
                    data.mnth, data.holiday,
                    data.weekday, data.workingday,
                    data.weathersit, data.temp,
                    data.atemp, data.hum, data.windspeed,
                    data.cnt.alias('label')
                    )
# data2.show()

train, test = data2.randomSplit([0.7,0.3])

# train.show(5)
# test.show(5)

assembler = VectorAssembler().setInputCols([
                    'season',
                    'yr',
                    'mnth',
                    'holiday',
                    'weekday',
                    'workingday',
                    'weathersit',
                    'temp',
                    'atemp',
                    'hum',
                    'windspeed'])\
    .setOutputCol('features')

train01 = assembler.transform(train)

# train01.limit(5).show(truncate=False)

train02 = train01.select("features","label")
train02.limit(5).show(truncate=False)

lr = LinearRegression()
model = lr.fit(train02)

train03 = model.transform(train02)
train03.limit(5).show(truncate=False)

test01 = assembler.transform(test)
test02 = test01.select('features', 'label')
test03 = model.transform(test02)

test03.limit(3).show(truncate=False)

evaluator = RegressionEvaluator()
print(evaluator.evaluate(test03, 
                   {evaluator.metricName: "r2"}) 
      )
print(evaluator.evaluate(test03, 
                   {evaluator.metricName: "mse"})
      )
print(evaluator.evaluate(test03, 
                   {evaluator.metricName: "rmse"})
      )
print(evaluator.evaluate(test03, 
                   {evaluator.metricName: "mae"})
      )