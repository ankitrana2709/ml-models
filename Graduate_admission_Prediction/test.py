#create a sparksession
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
spark = SparkSession.builder.appName("graduate").getOrCreate()

#create a spark dataframe
file = 'Admission_Predict_Ver1.1.csv'
df = spark.read.csv(file, header=True, inferSchema=True)

#drop the unnecessary column
df = df.drop('Serial No.')
df.show()

from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=['GRE Score', 'TOEFL Score', 'CGPA'], outputCol='features')

#display dataframe
output_df = assembler.transform(df)
output_df.show()
#import Linearregression and create final data
from pyspark.ml.regression import LinearRegression
from pyspark.sql.functions import *
final_data = output_df.select('features', 'Chance of Admit')