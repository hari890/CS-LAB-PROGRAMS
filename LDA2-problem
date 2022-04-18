import numpy as np 
import math

obs=np.array([2.81,5.46])#observations to be classified

A=np.array([[2.95,6.63],[2.53,7.79],[3.57,5.65],[3.16,5.47]])
print('A Matrix is:\n {}'.format(A))
A=A.mean(axis=0)
print('Mean of A:\n {}'.format(A))


B=np.array([[2.58,4.46],[2.16,6.22],[3.27,3.52]])
print('B Matrix is:\n {}'.format(B))
B=B.mean(axis=0)
print('Mean of B:\n {}'.format(B))


FM=np.array([[2.95,6.63],[2.53,7.79],[3.57,5.65],[3.16,5.47],[2.58,4.46],[2.16,6.22],[3.27,3.52]])
print('The Full Matrix is :\n {}'.format(FM))
print('Mean of Full Matrix:\n {}'.format(np.mean(FM,axis=0)))
FM = (FM - np.mean(FM, axis=0))
print('The Scaled Matrix is :\n {}'.format(FM))


PoolCov=np.dot(np.transpose(FM),FM)/7
print('The Variance Covariance Matrix is :\n {}'.format(np.round(PoolCov,2)))

pcinv=np.linalg.inv(PoolCov)
print('The inverse of Pooled Covariance Matrix is :\n {}'.format(pcinv))

mu1=np.transpose(A)
mu2=np.transpose(B)

f1=np.dot(np.dot(mu1,pcinv),np.transpose(obs))-(0.5*(np.dot(np.dot(mu1,pcinv),np.transpose(mu1))))+(math.log(4/7))
f2=np.dot(np.dot(mu2,pcinv),np.transpose(obs))-(0.5*(np.dot(np.dot(mu2,pcinv),np.transpose(mu2))))+(math.log(3/7))
print('\n\nf1 ={}'.format(np.round(f1,2)))
print('f2 ={}'.format(np.round(f2,2)))
