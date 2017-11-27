import numpy as np

a = np.ones(shape=[2,4,3])
print a

b = np.zeros(shape=[2,4,3])
print b

c = np.concatenate((a,b),axis=0)
print c,c.shape

d = np.concatenate((a,b),axis=1)
print d,d.shape


e = np.concatenate((a,b),axis=2)
print e,e.shape

a1 = e[:,:,0:3]
b1 = e[:,:,3:6]

print a1, a1.shape
print b1, b1.shape