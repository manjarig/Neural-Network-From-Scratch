import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix

def main():
    np.random.seed(0)
    #Get data info
    Xtrain,Xtest,Ytrain,Ytest=get_data_info()
    #V: Weight Matrix from input layer to hidden layer   an: pre synaptic hidden layer   Zn: post synaptic hidden layer 
    #W:Weight matrix from hidden layer to output layer   bn: pre synaptic output layer   Yn : post synaptic output layer
    V,W,an,Zn,bn,Yn=neuralmodel(Xtrain,Ytrain)
    evaluationmetrics(Ytrain,Yn)
    ################### RUN TEST DATA ##############################
    print("------------------------------------------------------------")
    print("TEST DATA")
    an,Zn,bn,Yn=forwardpropagation(V,W,Xtest)
    cost=calculate_cost(Yn,Ytest)
    print("Cost of test data is: "+str(cost))
    evaluationmetrics(Ytest,Yn)

def neuralmodel(Xtrain,Ytrain):
    V,W=initialise_parameter(Xtrain)
    iterations=0
    cost = 0
    previous_cost = 0
    while(previous_cost - cost > 0.000001 or iterations <2):
        iterations += 1
        previous_cost = cost
        an,Zn,bn,Yn=forwardpropagation(V,W,Xtrain)
        cost=calculate_cost(Yn,Ytrain)
        # print(cost)
        dW1,DW2=backwardpropagation(V,W,an,Zn,bn,Yn,Xtrain,Ytrain)
        V,W=updategradients(V,W,dW1,DW2)
    print("TRAIN DATA")
    print("Cost of train data is: "+str(cost))
    print("Number of epochs: "+str(iterations))
    return V,W,an,Zn,bn,Yn

def get_data_info():    
    df=pd.read_csv("iris.data.csv",header=None)
    df = df[df[4]!="Iris-setosa"]
    df[4] = [0 if x =='Iris-virginica' else 1 for x in df[4]] 
    df_norm = (df.iloc[:,1:5] - df.iloc[:,1:5].mean()) / (df.iloc[:,1:5].std())
    # Add bias
    df_norm.insert(0,'bias',1)
    l_col=df_norm.shape[1]
    X_df=df_norm.iloc[:,0:l_col]
    Y_df=df.iloc[:,-1]  
    Xtrain_df,Xtest_df,Ytrain_df,Ytest_df=train_test_split(X_df,Y_df,test_size=0.2, shuffle=True, random_state=1)  
    Xtrain=np.array(Xtrain_df)
    Xtest=np.array(Xtest_df)
    Ytrain=np.array(Ytrain_df)
    Ytest=np.array(Ytest_df)
    return Xtrain,Xtest,Ytrain,Ytest

def initialise_parameter(Xtrain):
    n_h=5
    n_y=1
    Xtrain=np.transpose(Xtrain)
    n_x=Xtrain.shape[0]  
    print("n_x")
    print(n_x)
    V = np.random.randn(n_h, n_x)   *0.01 # random weights between -1 to +1 and n_x contains bias too(already added in the function get_data_info)
    W = np.random.randn(n_y, n_h+1) *0.01
    print(V.shape)
    return V,W

def forwardpropagation(V,W,Xtrain):
    an = np.dot(V,np.transpose(Xtrain))
    Zn=sigmoid(an)

    Zn=np.insert(Zn,0,1,axis=0)
    bn=np.dot(W, Zn)
    Yn = sigmoid(bn)

    return an,Zn,bn,Yn

def sigmoid(Z1):
    return 1/(1+np.exp(-Z1))

def calculate_cost(Yn,Ytrain):
    N=Ytrain.shape[0]
    prob=np.multiply(np.log(Yn), Ytrain) + np.multiply((1 - Ytrain), np.log(1 - Yn)) ## cross entropy  for classification
    cost=-np.sum(prob)/N
    return cost

def backwardpropagation(V,W,an,Zn,bn,Yn,Xtrain,Ytrain):
    error=Yn-Ytrain
    delta_njV=np.dot(np.transpose(W),error)*Zn*(1-Zn)
    DW1=np.dot(delta_njV[1:],Xtrain)
    DW2=np.dot(error,np.transpose(Zn))
    return DW1,DW2

def updategradients(V,W,dW1,DW2):
    alpha =0.01
    V = V - alpha * dW1
    W = W - alpha * DW2
    return V,W

def evaluationmetrics(Actual,predicted):
    Lpredicted=len(predicted[0])
    Actual=Actual.reshape(len(Actual),1)
    predicted[predicted>=0.5]=1
    predicted[predicted<0.5]=0
    predicted=predicted.reshape(Lpredicted,1)
    conf_matrix=confusion_matrix(Actual,predicted, labels=None, sample_weight=None, normalize=None)
    print("Confusion Matrix")
    print(conf_matrix)
    total=sum(sum(conf_matrix))
    sensitivity1 = conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1])
    print('Sensitivity : ', sensitivity1 )
    specificity1 = conf_matrix[1,1]/(conf_matrix[1,0]+conf_matrix[1,1])
    print('Specificity : ', specificity1)
    accuracy1=(conf_matrix[0,0]+conf_matrix[1,1])/total
    print ('Accuracy : ', accuracy1*100)
    
if __name__ == "__main__":
    main()

   
