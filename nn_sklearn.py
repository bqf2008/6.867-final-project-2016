from sklearn.neural_network import MLPClassifier
from numpy import *

#Control Variables
n_UserAttr = 0 # number of user variables
#Design = 1 # include Design attributes or not 1-Yes, 0-No
itr = 10
itr2 = 5
nNeuron = 6

# load data from csv files
print "===== import data ====="
data = loadtxt('data/Data_E&W_group.txt')
Input = data[:,0:0]
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
    
    AccuracyTrain_sameData = zeros(itr2)
    AccuracyValidate_sameData = zeros(itr2)

    for j in range(itr2):
        #W,B = NN_BackProp(X_train,Y_train,k,N_nodes,lmbda,convergeCriterion)
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)

        clf.fit(X_train,Choice_train)

        def predictNN(X):
            Px = NN_FeedForward(X,W,B,k)
            return Px

            #print "===== Predict Training ====="
        predict = clf.predict_proba(X_train)
        Y_train_predict = predict.argmax(axis=1).reshape([nTrain,1])
        wrongNum = sum((Y_train_predict - Choice_train)!=0)
        accuracyTrain = 1 - wrongNum*1.0/nTrain
            #print 'Training Accuracy is: ',accuracyTrain
            #wrong = where((Y_train_predict - Y_train)!=0)
            #print wrong


            #print "===== Predict Validation ====="
        predict = clf.predict_proba(X_validate)
        Y_validate_predict = predict.argmax(axis=1).reshape([nValidate,1])
        wrongNum = sum((Y_validate_predict - Choice_validate)!=0)
        accuracyValidate = 1- wrongNum*1.0/nValidate
            #print 'Validation Accuracy is: ',accuracyValidate
            #wrong = where((Y_validate_predict - Y_validate)!=0)
            #print wrong

        AccuracyTrain_sameData[j] = accuracyTrain
        AccuracyValidate_sameData[j] = accuracyValidate

    AccuracyTrain[i] = AccuracyTrain_sameData[AccuracyValidate_sameData.argmax()]
    AccuracyValidate[i] = AccuracyValidate_sameData.max()
    print 'Training Accuracy is: ',AccuracyTrain_sameData[AccuracyValidate_sameData.argmax()]
    print 'Validation Accuracy is: ',AccuracyValidate_sameData.max()
 


print 'Average Train Accuracy is: '
print average(AccuracyTrain),'+-',std(AccuracyTrain)
print 'Average Validate Accuracy is: '
print average(AccuracyValidate),'+-',std(AccuracyValidate)
print
print 'Corresponding Train Accuracy is:'
print AccuracyTrain[AccuracyValidate.argmax()]
print 'Best Validate Accuracy is:'
print AccuracyValidate.max()

