from numpy import *
from nn_functions import NN_BackProp, NN_FeedForward


# Control Variables
n_UserAttr = 1 # number of user variables
#Design = 1 # include Design attributes or not 1-Yes, 0-No
itr = 3
nNeuron = 6

# load data from csv files
print "===== import data ====="
data = loadtxt('data/Data_E&W_group.txt')
Input = data[:,1:2]
datax = concatenate((Input,data[:,6:14]),axis=1)
datax = (datax-average(datax,axis=0))/(datax.max(0)-datax.min(0)) # normalize
data = concatenate((datax,data[:,-1:]),axis=1)

AccuracyTrain = zeros(itr)
AccuracyValidate = zeros(itr)

for i in range(itr):

    random.shuffle(data)    

    nData = shape(data)[0]
    nTrain = nData/2
    nValidate = nData - nTrain

    dataTrain = data[0:nTrain]
    dataValidate = data[nTrain:nData]
    
###### start resampling ####
##    u, indices = unique(dataTrain[:,-1], return_inverse=True)
##
##    dataTrain1 = dataTrain[indices==0]
##    dataTrain2 = dataTrain[indices==1]
##    dataTrain3 = dataTrain[indices==2]
##    dataTrain4 = dataTrain[indices==3]
##    
##    count = bincount(indices)
##    classSize = count.max()
##    classSize1 = shape(dataTrain1)[0]
##    classSize2 = shape(dataTrain2)[0]
##    classSize3 = shape(dataTrain3)[0]
##    classSize4 = shape(dataTrain4)[0] 
##    
##    dataTrain1_resample = dataTrain1[random.choice(classSize1,classSize-classSize1)]
##    dataTrain2_resample = dataTrain2[random.choice(classSize2,classSize-classSize2)]
##    dataTrain3_resample = dataTrain3[random.choice(classSize3,classSize-classSize3)]
##    dataTrain4_resample = dataTrain4[random.choice(classSize4,classSize-classSize4)]
##
##    dataTrain = concatenate((dataTrain,dataTrain1_resample,dataTrain2_resample,dataTrain3_resample,dataTrain4_resample),axis=0)
##
##    nTrain = shape(dataTrain)[0]
###### end resampling ####

    
    X_train = dataTrain[:,0:-1]
    Choice_train = dataTrain[:,-1:] - 1
    Y_train = concatenate(((Choice_train == 0)*1.,(Choice_train == 1)*1.,(Choice_train == 2)*1.,(Choice_train == 3)*1.),axis=1)
   
    X_validate = dataValidate[:,0:-1]
    Choice_validate = dataValidate[:,-1:] - 1
    Y_validate = concatenate(((Choice_validate == 0)*1.,(Choice_validate == 1)*1.,(Choice_validate == 2)*1.,(Choice_validate == 3)*1.),axis=1)


    # define NN structure
    k = 4 # number of output classes
    N_nodes = array([nNeuron]) # nunber of nodes in each hidden layer

    # train neural networks
    #print '====== Training ======'
    lmbda = 1
    convergeCriterion = 1e-6
    #W,B = NN_BackProp(X_train,Y_train,k,N_nodes,lmbda,Epoch)
    W,B = NN_BackProp(X_train,Y_train,k,N_nodes,lmbda,convergeCriterion)

    def predictNN(X):
        Px = NN_FeedForward(X,W,B,k)
        return Px


    #print "===== Predict Training ====="
    predict = predictNN(X_train)
    Y_train_predict = predict.argmax(axis=1).reshape([nTrain,1])
    wrongNum = sum((Y_train_predict - Choice_train)!=0)
    accuracyTrain = 1 - wrongNum*1.0/nTrain
    print 'Training Accuracy is: ',accuracyTrain
    #wrong = where((Y_train_predict - Y_train)!=0)
    #print wrong


    #print "===== Predict Validation ====="
    predict = predictNN(X_validate)
    Y_validate_predict = predict.argmax(axis=1).reshape([nValidate,1])
    wrongNum = sum((Y_validate_predict - Choice_validate)!=0)
    accuracyValidate = 1- wrongNum*1.0/nValidate
    print 'Prediction Accuracy is: ',accuracyValidate
    #wrong = where((Y_validate_predict - Y_validate)!=0)
    #print wrong

    AccuracyTrain[i] = accuracyTrain
    AccuracyValidate[i] = accuracyValidate

    if (i % 25) ==0:       
        print i


print 'Average Train Accuracy is: '
print average(AccuracyTrain),'+-',std(AccuracyTrain)
print 'Average Validate Accuracy is: '
print average(AccuracyValidate),'+-',std(AccuracyValidate)
print
print 'Corresponding Train Accuracy is:'
print AccuracyTrain[AccuracyValidate.argmax()]
print 'Best Validate Accuracy is:'
print AccuracyValidate.max()




















