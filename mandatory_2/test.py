import numpy as np
import matplotlib.pyplot as plt 

def f(x):
    return np.power(2,1-x)

def g(x):
    return 1 + np.power(-1,x)


a = np.arange(1,40).astype('float')

print(a)
print(np.sum(g(a)))
plt.plot(a,g(a))
plt.show()
#plt.plot(a,f(a))
#plt.plot(a,np.ones(len(a))*1/8)
#plt.show()
