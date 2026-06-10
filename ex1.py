#solving XOR problem using deep neural networks
import numpy as np
import matplotlib.pyplot as plt
import sys
X=np.array([[0,1],[1,0],[1,1],[0,0]])
y=np.array([[1],[1],[0],[0]])
i_units=2
h_units=2
o_units=1
learning_rate=0.01
reg_param=0
max_iter=5000
m=4
np.random.seed(1)
W1=np.random.normal(0,1,(h_units,i_units))
W2=np.random.normal(0,1,(o_units,h_units))
B1=np.random.random((h_units,1))
B2=np.random.random((o_units,1))
def sigmoid(z,derv=False):
    if derv:
        return z*(1-z)
    return 1/(1+np.exp(-z))
def forword(x,predict=False):
    a1=x.reshape(x.shape[0],1)
    z2=W1.dot(a1)+B1
    a2=sigmoid(z2)
    z3=W2.dot(a2)+B2
    a3=sigmoid(z3)
    if predict:
        return a3
    return a1,a2,a3
cost=np.zeros((max_iter,1))
def train(_W1,_W2,_B1,_B2):
    for i in range(max_iter):
        c=0
        dW1=0
        dW2=0
        dB1=0
        dB2=0
        for j in range(m):
            sys.stdout.write(f"\rIteration:{i+1} and {j+1}")
            a0=X[j].reshape(X[j].shape[0],1)
            z1=_W1.dot(a0)+_B1
            a1=sigmoid(z1)
            z2=_W2.dot(a1)+_B2
            a2=sigmoid(z2)
            
            dz2=a2-y[j]
            dW2+=dz2*a1.T
            dB2+=dz2
            dz1=np.multiply((_W2.T * dz2),sigmoid(a1,derv=True))
            dW1+=dz1.dot(a0.T)
            dB1+=dz1
            
            c+=-(y[j]*np.log(a2))-((1-y[j])*np.log(1-a2))
            sys.stdout.flush()
            
        _W1=_W1-learning_rate*(dW1/m)+(reg_param/m)*_W1
        _W2=_W2-learning_rate*(dW2/m)+(reg_param/m)*_W2
        _B1=_B1-learning_rate*(dB1/m)
        _B2=_B2-learning_rate*(dB2/m)
            
        cost[i]=(c/m)+(
            (reg_param/(2*m))*(
                np.sum(np.power(_W1,2))+
                np.sum(np.power(_W2,2))
            )
        )
    return _W1,_W2,_B1,_B2
W1,W2,B1,B2=train(W1,W2,B1,B2)
plt.plot(range(max_iter),cost)        
plt.xlabel("Iterations")
plt.ylabel("Cost")  
plt.title("Training cost over iterations")
plt.show()        