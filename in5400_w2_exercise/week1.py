import numpy as np
import torch as to

def loops(X, T):
    d = np.empty((len(X[0,:]), len(T[0,:])))
    print(d.shape)
    for i in range(len(X[0,:])):
        for j in range(len(T[0,:])):
            d[i,j]= np.linalg.norm(X[i,:] - T[j,:])

    return d

def numpy_solution(X,T): #this is not correct is seems
    d = np.empty((len(X[0,:]), len(T[0,:])))
    d = (X**2).sum(axis = 1, keepdims=True) +\
        (T**2).sum(axis=1)+\
        2*X @ T.T

    return d

def pytorch_attempt(X,T):
    X = to.tensor(X)
    T = to.tensor(T)
    d = to.sum(

    # torch.linalg.norm(X.sub(T))

D = 5
N = 10
P = 5

X = np.random.randint(10,size=(N,D))
T = np.random.randint(10,size=(P,D))

# d = numpy_solution(X,T)
print(d)

# d = loops(X,T)
print(d)
