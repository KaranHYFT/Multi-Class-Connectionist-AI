import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class NeuralNetwork:
    def __init__(self,x,y,learnrate,epoch):
        self.input      = x                 
        self.y          = y
        self.learnrate  = learnrate
        self.epoch      = epoch
        self.weights1   = np.random.rand(self.input.shape[1],6) 
        self.weights2   = np.random.rand(6,1)
        self.outcome    = np.zeros(self.y.shape)
        self.costDataList   = []

    # Calculating sigmoid
    @staticmethod    
    def sigmoidVal(x):
        return 1.0 / (1.0 + np.exp(-x))
    
    # Calculating Derivative of Sigmoid x(1-x)
    @staticmethod    
    def sigmoidDerivative(x): return x * (1 - x)      
    
    # Calculating cost 
    @staticmethod
    def findVals(targetData,outputData):
        return 0.5*np.sum(np.square(np.subtract(targetData,outputData)))
    
    # Apply Sigmoid on Layer and output
    def forwardMovement(self):
        self.HiddenLayer = self.sigmoidVal(np.dot(self.input, self.weights1))
        self.outcome = self.sigmoidVal(np.dot(self.HiddenLayer, self.weights2))

    def backpropogation(self):
        # derivative of the cost function 
        WeightVal2 = np.dot(self.HiddenLayer.T, ((self.y - self.outcome) * self.sigmoidDerivative(self.outcome)))
        WeightVal1 = np.dot(self.input.T,  (np.dot((self.y - self.outcome) * self.sigmoidDerivative(self.outcome), self.weights2.T) * self.sigmoidDerivative(self.HiddenLayer)))

        # Updating weights
        self.weights1 += WeightVal1 * self.learnrate
        self.weights2 += WeightVal2 * self.learnrate     
        
    # Training in data set    
    def dataTraining(self):
        for i in range (self.epoch):
            self.forwardMovement()
            self.backpropogation()
            self.costDataList.append(self.findVals(self.y,self.outcome))

    # Testing in data set        
    def dataPrediction(self,inputData):
        self.input=inputData
        self.forwardMovement()
        return self.outcome
            
              

if __name__ == "__main__":

    #Reading CSV file of Training
    data = pd.read_csv("IRIS_TrainData.csv",header=0)
    Trainingdata=[]
    ClassValue=[]

    #adding data to List for sepal/petal length/width and its associated type of iris flower
    for i in range(0,len(data)):
        Trainingdata.append([data.values[i,0],data.values[i,1],data.values[i,2],data.values[i,3]])
        if data.values[i,4]=='setosa':
            ClassValue.append([0.0001])
        elif data.values[i,4]=='versicolor':
            ClassValue.append([0.6666])
        else :
            ClassValue.append([0.9999])
    #Normalising using sklearn
    scalerData = StandardScaler()
    scalerData.fit(Trainingdata)
    scalerTrainingData=scalerData.transform(Trainingdata)
    

    network = NeuralNetwork(x=scalerTrainingData,y=np.array(ClassValue),learnrate=0.3,epoch=2000)
    #Data Training on Training Data
    network.dataTraining()

    # Reading data for Testing
    testdata = pd.read_csv("IRIS_testData.csv",header=0)
    testData=[]

    #Normalising the inputs of test data
    for i in range(0,len(testdata)):
        testData.append([testdata.values[i,0],testdata.values[i,1],testdata.values[i,2],testdata.values[i,3]])
    
    # Data Normalizing  
    xtest=scalerData.transform(testData)
    out=network.dataPrediction(xtest)
    
    ##Labeling the test data based on the output of the Neuralnet
    irisType=[]
    print("Output :")
    print("\nsepal_length  sepal_width  petal_length  petal_width  Output  IrisType")
    for i in range(0,len(xtest)):
        if(out[i]<0.2):
            irisType.append('setosa')
        elif(out[i]>=0.2 and out[i]<=0.8):
            irisType.append('versicolor')
        else:
            irisType.append('virginica')
        print("       ",testData[i][0],"        ",testData[i][1],"       ",testData[i][2],"      ",testData[i][3],"  ",out[i]," ",irisType[i])