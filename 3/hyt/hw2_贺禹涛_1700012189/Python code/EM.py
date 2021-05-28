import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg
from scipy.stats import norm
from pandas import DataFrame
import pandas as pd
import random
import math

p,K,N=10,5,100
THETA=np.zeros(K+1)
for k in range(1,K+1):
  THETA[k]=1.0/k
SIGMA=np.ones(K+1)

def softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x

  
def syn_data():
  x=np.random.multivariate_normal(np.zeros(p),np.identity(p),N)
  z=np.random.randint(low=1,high=K+1,size=N)
  y=np.zeros(N)
  for i in range(N):
    mu_i=THETA[z[i]]*sum(x[i])
    sigma_i=SIGMA[z[i]]
    y[i]=np.random.normal(mu_i,sigma_i)  
  
  return x,z,y

X,Z,Y=syn_data()  
def Estep(theta,sigma):
  phi=np.zeros([N,K+1])
  C=np.zeros(N)
  for n in range(N):
    for k in range(1,K+1):
      phi[n,k]=1/K*norm.pdf(x=Y[n],loc=np.matmul(theta[k,:],X[n]),scale=sigma[k])
    C[n]=sum(phi[n])
    phi[n]=phi[n]/C[n]
  
  return phi,C
  
def Mstep(phi,theta,sigma):
  for k in range(1,K+1):
    rhs=np.zeros(p)
    lhs=np.zeros([p,p])
    for i in range(p):
      rhs[i]=sum(X[:,i]*phi[:,k]*Y)
      for j in range(p):
        lhs[i,j]=sum(X[:,i]*X[:,j]*phi[:,k])
    #solve lhs*theta[k]=rhs
    theta[k]=linalg.solve(lhs,rhs)
    
  for k in range(1,K+1):
    tmp=Y-np.sum(theta[k]*X,axis=1)
    sigma2=sum(phi[:,k]*tmp**2)
    #for n in range(N):
    #  sigma2+=phi[n,k]*(Y[n]-np.matmul(theta[k,:],X[n]))**2
    sigma2=sigma2/sum(phi[:,k])
    sigma[k]=np.sqrt(sigma2)
 
def Marginal(phi,C):
  ans=0
  for n in range(N):
    for k in range(1,K+1):
      ans+=phi[n,k]*np.log(C[n]) 
  return ans 
   
def EM():
  theta=np.random.uniform(-1,3,size=[K+1,p])
  sigma=np.random.uniform(1,5,K+1)
  theta_rec=np.zeros([100,K+1,p])
  sigma_rec=np.zeros([100,K+1])
  margin_rec=np.zeros(100)
  for t in range(100):
    phi,C=Estep(theta,sigma)
    margin_rec[t]=Marginal(phi,C)
    Mstep(phi,theta,sigma)
    theta_rec[t]=theta
    sigma_rec[t]=sigma
  
  return theta_rec,sigma_rec,margin_rec

  
theta_rec,sigma_rec,margin_rec=EM()
#plt.plot(np.sum(theta_rec[:,1,:],axis=1))
#plt.plot(sigma_rec[:,1])
plt.plot(margin_rec)
plt.xlabel('iteration time')
plt.ylabel('marginal log likelihood')
plt.show()
plt.figure(2)
plt.bar(np.arange(0,N),Y)
plt.figure(3)
plt.bar(np.arange(0,N),Z)
plt.show() 