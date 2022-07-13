# Imports the libraries required for analysing the data
from IPython.display import display
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.feature import Imputer
import pyspark
from pyspark.sql import SparkSession
import pandas
import pyspark.sql.functions as func 
import numpy
import matplotlib
import matplotlib.pyplot as plt
from pyspark.ml.classification import DecisionTreeClassifier, LinearSVC, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from functools import reduce

#Creates the spark session
spark = SparkSession.builder.getOrCreate()


class BigData:
    
    def __init__(self):
        self.summaryStats = [] # Declares a list for gathering the summary statistics
        pandas.set_option('display.max_rows', 500)
        pandas.set_option('display.max_columns', 500)
        pandas.set_option('display.width', 1000)
        
    
    def readData(self):
        self.df = spark.read.csv("nuclear_plants_small_dataset.csv",inferSchema=True,header=True) # Reads the data and creates a data frame
        self.dataColumns = self.df.columns[1:] # Specifies only the data columns 
    def readBigData(self):
        self.bigDf = spark.read.csv("nuclear_plants_big_dataset.csv",inferSchema=True,header=True) # Reads the data and creates a data frame
        self.bigDataColumns = self.bigDf.columns[1:] # Specifies only the data columns 
      
        
        
    def fillMissingValues(self):
        self.df.na.fill('Missing Value',subset=['Status']) # Any missing values in the status column with "Missing Value"
    
       
        
       #Creates an imputer using mean to fill any missing values of the numerical columns
        imputer = Imputer(
            inputCols= self.dataColumns,outputCols=self.dataColumns
        ).setStrategy("mean")
    
        #Changes the data frame to an imputed version
        self.df = imputer.fit(self.df).transform(self.df)
    
    
    
    #Displays the data frame    
    def displayDataFrame(self):
        self.df.show()
        
    

    
    def calculateSummaryStatistics(self):
    
        
        for i in self.dataColumns:
        
            #Appends summary statistics grouped by the status column
            #Uses the .agg function to aggregate the data
            self.summaryStats.append(self.df.groupby('Status').agg(func.min(self.df[i]).alias(i + ' Min'),
            func.max(self.df[i]).alias(i + ' Max'),
            func.mean(self.df[i]).alias(i + ' Mean'),
            func.stddev(self.df[i]**2).alias(i + ' Variance'),
            func.percentile_approx(i,0.5).alias(i + ' Median')).toPandas())

    #Uses a string indexer to replace the status column with status index   
    def stringIndexer(self):
        self.indexer = StringIndexer(inputCol="Status", outputCol="StatusIndex")
        self.indexedDf = self.indexer.fit(self.df).transform(self.df)
        
    #Creates a dataframe vector using a vector assembler on the indexed data frame
    def vectorAssembler(self):
        self.vectorCol = "corr_features"
        assembler = VectorAssembler(inputCols = self.dataColumns, outputCol=self.vectorCol)
        self.dfVector = assembler.transform(self.indexedDf).select(self.vectorCol,self.indexedDf[-1])
        
        
    #Creates a correlation matrix based on the data frame vector
    def correlationMatrix(self):
        matrix = Correlation.corr(self.dfVector, self.vectorCol)
        corNumpy = matrix.collect()[0][matrix.columns[0]].toArray() #Converts to numpy array
        
        corMatrix = corNumpy.reshape(12,12)
        
        self.pandasCorMatrix = pandas.DataFrame(data = corMatrix, columns = self.dataColumns,index = self.dataColumns) #Converts to pandas data fame to be displayed
        
        
        
        
    def displayCorrelationMatrix(self):

        display(self.pandasCorMatrix) #Displays correlation matrix as a pandas data frame
    
    
    
    def displaySummaryStatistics(self):
        
        for i in self.summaryStats: #Itterates over the summaryStats list an displays each element
            display(i)
  

    def plotData(self,status):
        
        pandasDf = self.df.toPandas() #Converts the data frame to a pandas data frame to create a box plot
        plot = pandasDf.where(pandasDf.Status == status)#Plot the data based on the status
        plot.boxplot(figsize = (30,20))

    def shuffleData(self): 
        self.train, self.test = self.dfVector.randomSplit([0.7,0.3]) #Assigns variables for testing and training from the data frame vector
        print("Training set count: {0}, Testing set count: {1}".format(self.train.count(),self.test.count())) #Displays the testing and training set count

    
    def decisionTree(self):
        self.decisionTreeClass = DecisionTreeClassifier(featuresCol="corr_features",labelCol="StatusIndex") #Creates a decision tree classifier
        self.decisionTreeClass = self.decisionTreeClass.fit(self.train) #Fits the decision tree class to the training data

    def neuralNetwork(self):
        layers = [12, 5, 12, 2] #Specifies the layers for the neural network
        self.multiLayerPerceptron = MultilayerPerceptronClassifier(layers=layers,maxIter=100,blockSize=128,seed=1234,featuresCol="corr_features",labelCol="StatusIndex")
        self.multiLayerPerceptron = self.multiLayerPerceptron.fit(self.train) #Fits the neural network to the training data

    def linearSupportVector(self):
        self.lsvc = LinearSVC(maxIter=10,regParam=0.1,featuresCol="corr_features",labelCol="StatusIndex")
        self.lsvc = self.lsvc.fit(self.train) #Fits the linear support vector to the training data

    
    #Evaluates each model displaying the Error rate, Sensitivity and Specificity  
    def evaluating(self,modelType):
        
        evaluator = MulticlassClassificationEvaluator(labelCol="StatusIndex", predictionCol="prediction") #Creates an evaluator

        #Checks the modelType passed in is a decision tree and evaluates against the testing set 
        if modelType == "decisionTree":
            prediction = self.decisionTreeClass.transform(self.test)
            accuracy = evaluator.evaluate(prediction, {evaluator.metricName: "accuracy"})
            sensitivity = evaluator.evaluate(prediction, {evaluator.metricName: "recallByLabel"})
            specificity = evaluator.evaluate(prediction, {evaluator.metricName: "precisionByLabel"})
            errorRate = 100 - accuracy * 100
            
            print("Accuracy of decision tree:" , accuracy * 100,"%")
            print("Sensitivity of decision tree:" , sensitivity * 100,"%")
            print("Error rate of decision tree:", errorRate,"%")
            print("Specificity of decision tree:" , specificity * 100,"%\n")

        #Checks the modelType passed in is a neural network and evaluates against the testing set 
        if modelType == "neuralNetwork":
            
            prediction = self.multiLayerPerceptron.transform(self.test)
    
            accuracy = evaluator.evaluate(prediction, {evaluator.metricName: "accuracy"})
            sensitivity = evaluator.evaluate(prediction, {evaluator.metricName: "recallByLabel"})
            specificity = evaluator.evaluate(prediction, {evaluator.metricName: "precisionByLabel"})
            errorRate = 100 - accuracy * 100
            
            print("Accuracy of neural network:" , accuracy * 100,"%")
            print("Sensitivity of neural network:" , sensitivity * 100,"%")
            print("Error rate of neural network:", errorRate,"%")
            print("Specificity of neural network:" , specificity * 100,"%\n")


        #Checks the modelType passed in is a linear support vector and evaluates against the testing set 
        if modelType == "linearSupportVector":
            prediction = self.lsvc.transform(self.test)
           
            accuracy = evaluator.evaluate(prediction, {evaluator.metricName: "accuracy"})
            sensitivity = evaluator.evaluate(prediction, {evaluator.metricName: "recallByLabel"})
            specificity = evaluator.evaluate(prediction, {evaluator.metricName: "precisionByLabel"})
            errorRate = 100 - accuracy * 100
            
            
            print("Accuracy of linear support vector:" , accuracy * 100,"%")
            print("Sensitivity of linear support vector:" , sensitivity * 100,"%")
            print("Error rate of linear support vector:", errorRate,"%")
            print("Specificity of linear support vector:" , specificity * 100,"%\n")
            
            
    #Splits the data into chunks of length n        
    def chunkify(self,data,n):
        for i in range(0, len(data),n):
            yield (data[i:i + n])
        
   
    #Calculates the maximum
    def getMax(self,num1,num2):
        if num1 > num2:
            return num1
        else:
            return num2
    #Finds the max of two numbers for every chunk  
    def mapperMax(self,column):
        self.chunkify(column,10)
        return reduce(self.getMax,column)
    #Finds the min of two numbers for every chunk
    def mapperMin(self,column):
        self.chunkify(column,10)
        return reduce(self.getMin,column)
    
    #Finds the mean of two numbers for every chunk
    def mapperMean(self,column):
        self.chunkify(column,10)
        return reduce(self.getMean,column)
    #Calculates the minimum
    def getMin(self,num1,num2):
        if num1 < num2:
            return num1
        else:
            return num2
        
    #Calculates the mean
    def getMean(self,num1,num2):
        return (num1 + num2) / 2
    
    #Loops through the data and performs map reduce
    def loopAndMapReduce(self):
        self.maxList = []
        self.minList = []
        self.meanList = []
        for i in self.bigDataColumns:
            columnList = [row[0] for row in self.bigDf.select(i).collect()] #For every row
            self.maxList.append(self.mapperMax(columnList)) #Append the max value to a list
            self.minList.append(self.mapperMin(columnList)) #Append the min value to a list
            self.meanList.append(self.mapperMean(columnList)) #Append the mean value to a list
       
    #Displays the min,max and mean as a pandas data frame
    def displayMapReduce(self):
        array = numpy.array([self.maxList,self.minList,self.meanList]) #Creates a numpy array
        bigDataDf = pandas.DataFrame(data = array, index = ["Max", "Min", "Mean"], columns = self.bigDataColumns) #Creates a pandas data frame from the numpy array
        
        display(bigDataDf) #Displays the pandas data frame
            
            
        
        
        
        

# Creates class object and calls the methods in the appropriate order
b = BigData()

b.readData()
b.readBigData()
b.fillMissingValues()
b.displayDataFrame()



b.calculateSummaryStatistics()
b.displaySummaryStatistics()


b.stringIndexer()  
b.vectorAssembler()
b.correlationMatrix()
b.displayCorrelationMatrix()


maxList = []
minList = []
meanList = []


b.plotData("Normal")
b.plotData("Abnormal")


b.shuffleData() 
b.decisionTree()
b.neuralNetwork()
b.linearSupportVector()

b.evaluating("decisionTree")
b.evaluating("linearSupportVector")
b.evaluating("neuralNetwork")

b.loopAndMapReduce()
b.displayMapReduce()



