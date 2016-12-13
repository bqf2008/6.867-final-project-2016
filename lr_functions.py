import numpy as np

def f_ELR(X,n_UserAttr,design,Y,omega,r_lambda):
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
    Xu = np.concatenate((one,X[:,0:n_UserAttr]),axis=1)
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
    # Cross Entropy Loss
    LossCE = -np.sum(Y*np.log(P))
    # L2 regularization to Loss Function
    Elr = LossCE + r_lambda * np.sum(omega**2)
    return Elr


def df_ELR(X,n_UserAttr,design,Y,omega,r_lambda):
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
    Xu = np.concatenate((one,X[:,0:n_UserAttr]),axis=1)
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
    # direvative
    Delta = np.eye(4)
    delta1 = Delta[0]
    delta2 = Delta[1]
    delta3 = Delta[2]
    delta4 = Delta[3]
    P1 = P[:,0].reshape(n,1)
    P2 = P[:,1].reshape(n,1)
    P3 = P[:,2].reshape(n,1)
    P4 = P[:,3].reshape(n,1)
    d_omega_u1 = np.sum(P1*Xu*np.sum(Y*(delta1[np.newaxis,:] - P)/P,axis=1).reshape(n,1),axis=0)
    d_omega_u2 = np.sum(P2*Xu*np.sum(Y*(delta2[np.newaxis,:] - P)/P,axis=1).reshape(n,1),axis=0)
    d_omega_u3 = np.sum(P3*Xu*np.sum(Y*(delta3[np.newaxis,:] - P)/P,axis=1).reshape(n,1),axis=0)
    d_omega_d = np.sum(P1*Xd1*np.sum(Y*(delta1[np.newaxis,:] - P)/P,axis=1).reshape(n,1),axis=0) + \
                np.sum(P2*Xd2*np.sum(Y*(delta2[np.newaxis,:] - P)/P,axis=1).reshape(n,1),axis=0) + \
                np.sum(P3*Xd3*np.sum(Y*(delta3[np.newaxis,:] - P)/P,axis=1).reshape(n,1),axis=0) + \
                np.sum(P4*Xd4*np.sum(Y*(delta4[np.newaxis,:] - P)/P,axis=1).reshape(n,1),axis=0)
    #print np.shape(d_omega_d)
    if design == 0:
        d_omega = -np.concatenate((d_omega_u1,d_omega_u2,d_omega_u3),axis=0)
    elif design == 1:
        d_omega = -np.concatenate((d_omega_u1,d_omega_u2,d_omega_u3,d_omega_d[0:1]),axis=0) 
    elif design == 2:
        d_omega = -np.concatenate((d_omega_u1,d_omega_u2,d_omega_u3,d_omega_d[1:]),axis=0) 
    elif design == 3:
        d_omega = -np.concatenate((d_omega_u1,d_omega_u2,d_omega_u3,d_omega_d),axis=0) 


    dElr = d_omega + 2 * r_lambda * omega
    return dElr


def GD(X, n_UserAttr, design, Y,  f, fd, r_lambda, init, step, convergeCriterion):

    # load the objective function f, its derivative function f_d, and other parameters
    # calculate the minimum using gradient descent algorithm
    # return a tuple that contains the minimum value, the minimum location, and the step to converge
   
    omega_old = init
    E_old = f(X, n_UserAttr, design, Y, init, r_lambda)
#    print "Loss"
#    print E_old

    count = 1
    
    omega_new = omega_old - step * fd(X, n_UserAttr, design, Y, omega_old, r_lambda)
    E_new = f(X, n_UserAttr, design, Y, omega_new, r_lambda)

 #   track = np.append(omega_old, omega_new, 1)
    
    while(np.abs(E_new - E_old) > convergeCriterion and count < 5e5):
        count = count + 1
        #print count

        omega_old = omega_new
        E_old = E_new
        #print E_old
        
        omega_new = omega_old - step * fd(X, n_UserAttr, design, Y, omega_old, r_lambda)
        E_new = f(X, n_UserAttr, design, Y, omega_new, r_lambda)
        #print omega_new
        #print E_new

 #       track = np.append(track,omega_new,1)
            
    Omega = omega_new
    minE = E_new
    
    return (minE, Omega, count)

