from numpy import *
from lr_functions import *


# Control Variables
n_UserAttr = 1  # number of user variables
Design = 3 # include Design attributes or not  0-No,1-clarity,2-emotion,3-both
itr = 100 # iteration number

# load data from csv files
print 'n_UserAttr is:', n_UserAttr

print "===== import data ====="
data = loadtxt('data/Data_E&W_group.txt')
datax = data[:,0:14]
#datax = (datax-average(datax,axis=0))/(datax.max(0)-datax.min(0)) # normalize
datax = (datax-datax.min(0))/(datax.max(0)-datax.min(0)) # normalize
data = concatenate((datax,data[:,-1:]),axis=1)
print shape(data)

AccuracyTrain = zeros(itr)
AccuracyValidate = zeros(itr)
if Design == 0:
    Omega = zeros([itr,(n_UserAttr+1)*3])
elif Design == 1:
    Omega = zeros([itr,(n_UserAttr+1)*3+1])
elif Design == 2:
    Omega = zeros([itr,(n_UserAttr+1)*3+1])
elif Design == 3:
    Omega = zeros([itr,(n_UserAttr+1)*3+2])


for i in range(itr):

    random.shuffle(data)
    Input = data[:,4:5]
#    Input = concatenate((data[:,1:3],data[:,3:6]),axis=1) # need to manually change input
    Choice = data[:,-1].reshape([shape(data)[0],1]) - 1

    X = concatenate((Input,data[:,6:14]),axis=1)    
    Y = concatenate(((Choice == 0)*1.,(Choice == 1)*1.,(Choice == 2)*1.,(Choice == 3)*1.),axis=1)
    
    nData = shape(data)[0]
    nTrain = nData/2
    nValidate = nData - nTrain

    X_train = X[0:nTrain]
    X_validate = X[nTrain:nData]

    Y_train = Y[0:nTrain]
    Y_validate = Y[nTrain:nData]

    Choice_train = Choice[0:nTrain]
    Choice_validate = Choice[nTrain:nData]

    f = f_ELR
    fd = df_ELR
    r_lambda = 0.1
    step = 1e-2
    #init = ones((n_UserAttr+1)*3+2)
    init = random.normal(0, 1./((n_UserAttr+1)*3+2), [(n_UserAttr+1)*3+2])
    if Design == 0:
        init = random.normal(0, 1./((n_UserAttr+1)*3), [(n_UserAttr+1)*3])
    elif Design == 1:
        init = random.normal(0, 1./((n_UserAttr+1)*3+1), [(n_UserAttr+1)*3+1])
    elif Design == 2:
        init = random.normal(0, 1./((n_UserAttr+1)*3+1), [(n_UserAttr+1)*3+1])
    elif Design == 3:
        init = random.normal(0, 1./((n_UserAttr+1)*3+2), [(n_UserAttr+1)*3+2])           
    
    convergeCriterion = 1e-6
    minE, omega, count = GD(X_train, n_UserAttr,Design, Y_train, f, fd, r_lambda, init, step, convergeCriterion)
#    print "iterations = ",count
#    print "omega =",omega

    Omega[i] = omega
    

    def predictLR(X,n_UserAttr,design):
        n = np.shape(X)[0]
        # parameters
        omega_u1 = omega[0:(n_UserAttr+1)]
        omega_u2 = omega[(n_UserAttr+1):2*(n_UserAttr+1)]
        omega_u3 = omega[2*(n_UserAttr+1):3*(n_UserAttr+1)]
        if design == 0:
            omega_d = np.array([0,0])
        elif design == 1:
            omega_d = np.array([omega[3*(n_UserAttr+1)],0])
        elif design == 2:
            omega_d = np.array([0,omega[3*(n_UserAttr+1)]])
        elif design == 3:
            omega_d = omega[3*(n_UserAttr+1):]
        # inputs
        one = np.ones([n,1])
        Xu = concatenate((one,X[:,0:n_UserAttr]),axis=1)
        Xd1 = X[:,n_UserAttr:n_UserAttr+2]
        Xd2 = X[:,n_UserAttr+2:n_UserAttr+4]
        Xd3 = X[:,n_UserAttr+4:n_UserAttr+6]
        Xd4 = X[:,n_UserAttr+6:n_UserAttr+8]
        # utilities
        U1 = (np.dot(Xu,omega_u1) + np.dot(Xd1,omega_d)).reshape(n,1)
        U2 = (np.dot(Xu,omega_u2) + np.dot(Xd2,omega_d)).reshape(n,1)
        U3 = (np.dot(Xu,omega_u3) + np.dot(Xd3,omega_d)).reshape(n,1)
        U4 = np.dot(Xd4,omega_d).reshape(n,1)    
        U = np.concatenate((U1,U2,U3,U4),axis=1)
        # probabilities    
        P = np.exp(U)/np.sum(np.exp(U),axis=1).reshape(n,1)
        return P


#    print "===== Predict Training ====="
    predict = predictLR(X_train,n_UserAttr,Design)
    Y_train_predict = predict.argmax(axis=1).reshape([nTrain,1])
    wrongNum = sum((Y_train_predict - Choice_train)!=0)
    accuracyTrain = 1 - wrongNum*1.0/nTrain
    #print 'Accuracy is: ',accuracyTrain
    #wrong = where((Y_train_predict - Y_train)!=0)
    #print wrong


#    print "===== Predict Validation ====="
    predict = predictLR(X_validate,n_UserAttr,Design)
    Y_validate_predict = predict.argmax(axis=1).reshape([nValidate,1])
    wrongNum = sum((Y_validate_predict - Choice_validate)!=0)
    accuracyValidate = 1- wrongNum*1.0/nValidate
    #print 'Accuracy is: ',accuracyValidate
    #wrong = where((Y_validate_predict - Y_validate)!=0)
    #print wrong

    AccuracyTrain[i] = accuracyTrain
    AccuracyValidate[i] = accuracyValidate

    if (i % 25) == 0:
        print i

print 'Accuracy Train is: '
print average(AccuracyTrain),'+-',std(AccuracyTrain)
print
print 'Accuracy Validate is: '
print average(AccuracyValidate),'+-',std(AccuracyValidate)
print
print 'Average Omega is:'
print average(Omega,axis=0)
print 'Std Omega is:'
print std(Omega,axis=0)
