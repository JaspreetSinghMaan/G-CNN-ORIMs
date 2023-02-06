import tensorflow as tf
import numpy as np
import math

def PZM(data_tensor):
    feat = tf.concat([PZMs(data_tensor,0,0), PZMs(data_tensor,1,0), PZMs(data_tensor,1,1), 
                  PZMs(data_tensor,2,0), PZMs(data_tensor,2,1),PZMs(data_tensor,2,2), 
                  PZMs(data_tensor,3,0), PZMs(data_tensor,3,1), PZMs(data_tensor,3,2), PZMs(data_tensor,3,3),
                  PZMs(data_tensor,4,0), PZMs(data_tensor,4,1), PZMs(data_tensor,4,2), PZMs(data_tensor,4,3), 
                  #PZMs(data_tensor,4,4), PZMs(data_tensor,5,0), PZMs(data_tensor,5,1), PZMs(data_tensor,5,2), 
                  #PZMs(data_tensor,5,3), PZMs(data_tensor,5,4), PZMs(data_tensor,5,5), PZMs(data_tensor,6,0),
                  #PZMs(data_tensor,6,1), PZMs(data_tensor,6,2), PZMs(data_tensor,6,3), PZMs(data_tensor,6,4),
                  #PZMs(data_tensor,6,5), PZMs(data_tensor,6,6), PZMs(data_tensor,7,0), PZMs(data_tensor,7,1),
                  #PZMs(data_tensor,7,2), PZMs(data_tensor,7,3), PZMs(data_tensor,7,4), PZMs(data_tensor,7,5),
                  #PZMs(data_tensor,7,6), PZMs(data_tensor,7,7), PZMs(data_tensor,8,0) ,PZMs(data_tensor,8,1),
                  #PZMs(data_tensor,8,2), PZMs(data_tensor,8,3), PZMs(data_tensor,8,4) ,PZMs(data_tensor,8,5),
                  #PZMs(data_tensor,8,6), PZMs(data_tensor,8,7), PZMs(data_tensor,8,8) ,PZMs(data_tensor,9,0),
                  #PZMs(data_tensor,9,1), PZMs(data_tensor,9,2), PZMs(data_tensor,9,3) ,PZMs(data_tensor,9,4),
                  #PZMs(data_tensor,9,5), PZMs(data_tensor,9,6), PZMs(data_tensor,9,7) ,PZMs(data_tensor,9,8),
                  #PZMs(data_tensor,9,9),
                  ],1)
    return feat

def PZMs(p,n,m):
  p = tf.transpose(p,(0,3,1,2))
  N = int(p.shape[2])
  x = np.arange(0,N,1)
  y = np.arange(0,N,1)
  D = N*np.sqrt(2)
  [X,Y] = np.meshgrid(x,y)
  R = np.sqrt((2.*X-N+1)**2+(2.*Y-N+1)**2)/D

  Theta = np.arctan2((2.*Y-N+1)/D, (2.*X-N+1)/D)
  Theta = Theta.transpose(1,0) 
  Theta = ((Theta<0.0)*(2.0*np.pi+Theta))+((Theta>=0.0)*(Theta))

  Rad = radialpoly(R,n,m)    # get the radial polynomial
  norm1 = (4.0*((n+1)/np.pi))/(D*D)
  ele2 = tf.cast(Rad, tf.complex64)*tf.cast(tf.exp(-1j*m*Theta), tf.complex64)

  Product = tf.cast(p,tf.complex64)*ele2
  Z = (tf.reduce_sum(tf.reduce_sum(Product,axis=3,keepdims=False),axis=2,keepdims=False))
  Z = norm1*Z
  A = tf.abs(Z)
  return A
  
def radialpoly(r,n,m):
  size_r = r.shape
  #print(size_r)
  rad = tf.zeros(size_r, tf.float32)                     # Initilization
  #print(rad.shape)
  s1 = np.int32((n-abs(m)))
  for s in range(s1+1):
      c = (-1)**s*math.factorial(2*n+1-s)/(math.factorial(s)*math.factorial(n+abs(m)+1-s)*math.factorial(n-abs(m)-s))
      rad = rad + c*r**(n-s)
  return rad