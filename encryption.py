'''
Created on Feb 7, 2019

@author: saurabh.chakraborty
'''
from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, IntegerType, StructType, StructField
import hashlib

def encrypt_value(mobno):
    sha_value = hashlib.sha256(mobno.encode()).hexdigest()
    return sha_value


conf = \
SparkConf().setMaster('local').setAppName('column_encryption')
sc = SparkContext(conf=conf)
sqlcontext = SQLContext(sc)

schema1 = StructType([
    StructField("cust_id", IntegerType(), True),
    StructField("mobno", StringType(), True),
    ])


data = sqlcontext.read.csv('sample.csv', header=True, schema=schema1)
data.show()

spark_udf = udf(encrypt_value, StringType())
data = data.withColumn('encrypted_value',spark_udf('mobno'))
data.show(truncate=False)


