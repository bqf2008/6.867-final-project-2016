from numpy import *

def ReLU(x):
    # activation function
    return x*(x>0)

def dReLU(x):
    # defivative of the activation function
    return (x>0)*1.0

def Softmax(x):
    # softmax
    e_x = exp(x - max(x))
    return e_x / e_x.sum(axis=0)


def NN_FeedForward(X,W,B,k):
    # feed forward nueral network
    l = shape(B)[0] - 1 # number of hidden layers of the network
    n = shape(X)[0] # number of data points
    
    a = X
    for i in range(l):
        w = W[i]
        b = B[i].flatten()
        z = dot(a,w) + b
        a = ReLU(z)

    w = W[-1]
    b = B[-1].flatten()
    z_out = dot(a,w) + b
    
    if size(shape(X)) == 2:
        P = zeros([n,k])        
        for i in range(n):
            P[i] = Softmax(z_out[i])
    elif size(shape(X)) == 1:
        P = Softmax(z_out)
            
    return P


def NN_BackProp(Xt,Yt,k,N_nodes,lmbda,convergeCriterion):
    # input: Xt,Yt - training data, Xv,Yv - validation data
    #        k - number of classes, N_nodes - number of nodes in each layer
    #        lmbda - learning rate, threshold - converge criteria
    d = shape(Xt)[1] # dimension of input data
    nTrain = shape(Xt)[0]
    l = size(N_nodes) # number of hidden layers

    DataT = concatenate((Xt,Yt),axis=1)
    
    # initialization
    n0 = d
    n1 = N_nodes[0]
    w = random.normal(0, 1./n0**0.5, [n0,n1])
    b = random.normal(0, 1./n1**0.5, [n1,1])
    W = [w]
    B = [b]
    for i in range(1,l):
        n0 = N_nodes[i-1]
        n1 = N_nodes[i]
        w = random.normal(0, 1./n0**0.5, [n0,n1])
        b = random.normal(0, 1./n1**0.5, [n1,1])
        W.append(w)
        B.append(b)
    n0 = N_nodes[-1]
    n1 = k
    w = random.normal(0, 1./n0**0.5, [n0,n1])
    b = random.normal(0, 1./n1**0.5, [n1,1])
    W.append(w)
    B.append(b)

    
    t = 0
    epoch = 0
    LossNew = 1e5
    LossOld = 0
    
    while (abs(LossNew - LossOld) > convergeCriterion) and (epoch < 5e2):
    #while epoch < MaxEpoch:
        # use certain number of training iterations
        epoch = epoch + 1
        random.shuffle(DataT)
        for i in range(nTrain):
            t = t+1
            
            data = DataT[i]
            x = DataT[i,0:d].reshape([d,1])
            y = DataT[i,d:d+k].reshape([k,1])
        
            # feed forward
            z = dot(W[0].T,x) + B[0]
            a = ReLU(z)
            Z = [z]
            A = [a]
            for i in range(1,l):
                z = dot(W[i].T,a) + B[i]
                a = ReLU(z)
                Z.append(z)
                A.append(a)
            z_out = dot(W[-1].T,a) + B[-1]
            
                
            # back propogation
            delta = Softmax(z_out) - y
            Delta = [delta]

            for i in arange(l-1,-1,-1):
                delta = dot(diag(dReLU(Z[i]).flatten()), dot(W[i+1],delta))
                Delta = [delta]+Delta
                    
            # stochastic gradient descent
            eta = 1./((t**0.5)*lmbda)
#            eta = 1./lmbda # fixed step size
            
            dw = dot(x,Delta[0].reshape([1,N_nodes[0]]))
            db = Delta[0]
            W[0] = W[0] - eta*dw
            B[0] = B[0] - eta*db
            for i in range(1,l):
                dw = dot(A[i-1],Delta[i].reshape([1,N_nodes[i]]))
                db = Delta[i]                
                W[i] = W[i] - eta*dw
                B[i] = B[i] - eta*db    
            dw = dot(A[-1],Delta[-1].reshape([1,k]))
            db = Delta[-1]
            W[-1] = W[-1] - eta*dw
            B[-1] = B[-1] - eta*db

        P = NN_FeedForward(Xt,W,B,k)
        #print 'P', P
        #print 'log(P)', log(P)
        LossOld = LossNew
        LossNew = -sum(Yt*log(P)*array([1,1,3,2]))
##        if (epoch %100) == 0:            
##            print 'epoch:', epoch
##        print 'Loss: ', LossNew

##    print 'Total epoch: ',epoch              
            
    return W,B



