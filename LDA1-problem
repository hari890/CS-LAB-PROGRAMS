import numpy as np 

A=np.array([[1,1300,2.7],[1,1260,3.7],[1,1220,2.9],[1,1180,2.5],[1,1060,3.9],[1,1140,2.1],[1,1100,3.5],[1,1020,3.3],[1,980,2.3],[1,940,3.1]])
print('Coeefficient Matrix is:\n {}'.format(A))

Y=np.array([[1],[1],[1],[1],[1],[0],[0],[0],[0],[0]])
print('Dependent Matrix is:\n {}'.format(Y))

AT=np.transpose(A)
betahat=np.dot(np.linalg.inv(np.dot(AT,A)),np.dot(AT,Y))
print('Regression Coefficients are:\n {}'.format(betahat))

yhat=np.dot(A,betahat)
print('Yhat Matrix is:\n{}'.format(yhat))

eps=Y-yhat
print('Error Matrix is :\n{}'.format(eps))

sse=np.dot(np.transpose(eps),eps)
print('Sum of squares due to error is :\n{}'.format(sse))

sst=np.dot(np.transpose(Y- np.mean(Y, axis=0)),Y-np.mean(Y,axis=0))
print('Sum of squares due to total is :\n{}'.format(sst))

ssr=sst-sse
print('Sum of squares due to Regression Model is :\n{}'.format(ssr))
