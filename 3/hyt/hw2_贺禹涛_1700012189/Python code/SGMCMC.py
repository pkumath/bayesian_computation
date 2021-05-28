import sys
sys.path.append('/raid/heyutao/code/hw/ITE/')

import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame
import pandas as pd
import random
import math
from ite.cost.x_analytical_values import analytical_value_d_kullback_leibler
from ite.cost.x_factory import co_factory

sth,sy=1,5

THETA=np.loadtxt('/raid/heyutao/code/hw/HMC.txt')
co = co_factory('BDKL_KnnK', mult=True)

def batch_grad_theta1(by,theta):
  return sum(by-theta[0]-theta[1]**2)/sy**2
  
def batch_grad_theta2(by,theta):
  return 2*theta[1]*sum(by-theta[0]-theta[1]**2)/sy**2
  
def p_grad_theta1(theta):
  return -theta[0]/sth**2
  
def p_grad_theta2(theta):
  return -theta[1]/sth**2
  
def SGLD(y,theta0,test_T):
  N=len(y)
  n=50
  T=100001
  dlist=[]
  #theta=np.array([-3.0,2.1])
  theta=np.array(theta0)
  param_rec=np.zeros([T,2])
  cnt=0
  for t in range(T):
    if(t%5000==0):
      print(t)
    ind=random.sample(range(N),n)
    g1=N/n*batch_grad_theta1(y[ind],theta)+p_grad_theta1(theta)
    g2=N/n*batch_grad_theta2(y[ind],theta)+p_grad_theta2(theta)
    et=1e-4/math.pow(t+1,1e-4)
    d1=np.random.normal(et/2*g1,et)
    d2=np.random.normal(et/2*g2,et)
    theta+=np.array([d1,d2])
    param_rec[t,:]=theta
    if(t==test_T[cnt]):
     d=co.estimation(THETA,param_rec[0:t,:]) 
     dlist.append(d)
     cnt+=1
  #return param_rec
  return dlist
  
#def SGHMC_step(theta,L,et):

def SGHMC(y,theta0,test_T):
  N=len(y)
  n=50
  T=100001
  dlist=[]
  C=20
  cnt=0
  theta=np.array(theta0)  
  param_rec=np.zeros([T,2])
  for t in range(T):
    if(t%5000==0):
      print(t)
    r=np.random.normal(0,1,2)
    L=np.random.randint(1,30)
    et=1e-3/math.pow(t+1,0.05)
    for i in range(L):
      theta=theta+et*r
      ind=random.sample(range(N),n)
      g1=N/n*batch_grad_theta1(y[ind],theta)+p_grad_theta1(theta)
      g2=N/n*batch_grad_theta2(y[ind],theta)+p_grad_theta2(theta)
      r=r+et*np.array([g1,g2])-et*C*r+np.random.normal(0,np.sqrt(2*C*et),2)
    param_rec[t,:]=theta
    if(t==test_T[cnt]):
     #print(t)
     d=co.estimation(THETA,param_rec[0:t,:]) 
     dlist.append(d)
     cnt+=1
  return dlist

def SGNHT(y0,theta0,test_T):
  N=len(y)
  n=50
  T=100001
  dlist=[]
  cnt=0
  A=100
  theta=np.array(theta0)
  r,E=np.random.normal(0,1,2),A
  param_rec=np.zeros([T,2])
  for t in range(T):
    if(t%5000==0):
      print(t)
    ind=random.sample(range(N),n)
    g1=N/n*batch_grad_theta1(y[ind],theta)+p_grad_theta1(theta)
    g2=N/n*batch_grad_theta2(y[ind],theta)+p_grad_theta2(theta)
    et=1e-3/math.pow(t+1,0.05)
    #et=1e-2
    r=r+et*np.array([g1,g2])-et*E*r+np.sqrt(2*A)*np.random.normal(0,np.sqrt(et),2)
    theta=theta+et*r
    E=E+et*(sum(r**2)/2-1)
    param_rec[t,:]=theta
    if(t==test_T[cnt]):
     d=co.estimation(THETA,param_rec[0:t,:]) 
     dlist.append(d)
     cnt+=1
  return param_rec, dlist
  
y=np.random.normal(1,sy,10000)
test_T=np.array([500,1000,5000,10000,50000,100000])
#d1=SGLD(y,[-3.0,2.2],test_T)
#plt.plot(test_T,d1,label='SGLD')
#d2=SGHMC(y,[-3.0,2.2],test_T)
#plt.plot(test_T,d2,label='SGHMC')
param_rec,d3=SGNHT(y,[-3.0,2.2],test_T)
plt.scatter(THETA[:,0],THETA[:,1],label='ground truth')
plt.scatter(param_rec[500:-1,0],param_rec[500:-1,1],label='SGNHT')
plt.xlabel('theta_1')
plt.ylabel('theta_2')
plt.legend()
plt.show()
plt.plot(test_T,d3,label='SGNHT')
plt.legend()
plt.show()

"""
param_rec1_1=SGLD(y,[-3.0,2.2])
param_rec1_2=SGLD(y,[-3.0,-2.2])
param_rec1=np.concatenate((param_rec1_1,param_rec1_2),axis=0)

d1=co.estimation(THETA,param_rec1_1)
d2=co.estimation(THETA,param_rec1_2)
d=co.estimation(THETA,param_rec1)
print(d1,d2,d)
#5.986220862726113 5.452054246356817 1.6756601303422338
plt.figure(1)
plt.scatter(THETA[:,0],THETA[:,1],label='ground truth')
plt.scatter(param_rec1_1[:,0],param_rec1_1[:,1],label='SGLD (start at (-3.0,2.2))')
plt.legend()
plt.xlabel('theta_1')
plt.ylabel('theta_2')
plt.figure(2)
plt.scatter(THETA[:,0],THETA[:,1],label='ground truth')
plt.scatter(param_rec1_2[:,0],param_rec1_2[:,1],label='SGLD (start at (-3.0,-2.2)')
plt.xlabel('theta_1')
plt.ylabel('theta_2')
plt.legend()
plt.figure(3)
plt.scatter(THETA[:,0],THETA[:,1],label='ground truth')
plt.scatter(param_rec1[:,0],param_rec1[:,1],label='SGLD (merge)')
plt.xlabel('theta_1')
plt.ylabel('theta_2')
plt.legend()
plt.show()
#time=np.array(range(len(param_rec1[5000:30000,0])))
#plt.plot(time,param_rec1[5000:30000,0],label='theta_1')
#plt.plot(time,param_rec1[5000:30000,1],label='theta_2')
#plt.xlabel('time')
#plt.ylabel('theta')
#plt.grid(True)
#plt.legend()
#plt.show()

#param_rec2=SGHMC(y,np.random.normal(0,3*sth,2))
param_rec2=SGHMC(y,[-3.0,2.1])
#d0=co.estimation(THETA,THETA)
d2=co.estimation(THETA,param_rec2)
#2.870535096648451
print(d2)
plt.scatter(THETA[:,0],THETA[:,1],label='ground truth')
plt.scatter(param_rec2[500:-1,0],param_rec2[500:-1,1],label='SGHMC')
plt.xlabel('theta_1')
plt.ylabel('theta_2')
plt.legend()
plt.show()

param_rec3=SGNHT(y,np.random.normal(0,1,2))
d3=co.estimation(THETA,param_rec3)
print(d3)
#7.445842099121757 4.12029770656326 7.095060082751999
plt.scatter(THETA[:,0],THETA[:,1],label='ground truth')
plt.scatter(param_rec3[500:-1,0],param_rec3[500:-1,1],label='SGNHT')
plt.xlabel('theta_1')
plt.ylabel('theta_2')
plt.legend()
plt.show()
"""  