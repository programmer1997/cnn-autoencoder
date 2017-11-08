from load import mnist
import numpy as np
import pylab
import matplotlib.pyplot as plt

import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool


 
np.random.seed(10)
batch_size = 128
noIters = 100

def init_weights_bias4(filter_shape, d_type):
    fan_in = np.prod(filter_shape[1:])
    fan_out = filter_shape[0] * np.prod(filter_shape[2:])
     
    bound = np.sqrt(6. / (fan_in + fan_out))
    w_values =  np.asarray(
            np.random.uniform(low=-bound, high=bound, size=filter_shape),
            dtype=d_type)
    b_values = np.zeros((filter_shape[0],), dtype=d_type)
    return theano.shared(w_values,borrow=True), theano.shared(b_values, borrow=True)

def init_weights_bias2(filter_shape, d_type):
    fan_in = filter_shape[1]
    fan_out = filter_shape[0]
     
    bound = np.sqrt(6. / (fan_in + fan_out))
    w_values =  np.asarray(
            np.random.uniform(low=-bound, high=bound, size=filter_shape),
            dtype=d_type)
    b_values = np.zeros((filter_shape[1],), dtype=d_type)
    return theano.shared(w_values,borrow=True), theano.shared(b_values, borrow=True)

def set_weights_bias4(filter_shape, d_type, w, b):
    fan_in = np.prod(filter_shape[1:])
    fan_out = filter_shape[0] * np.prod(filter_shape[2:])
     
    bound = np.sqrt(6. / (fan_in + fan_out))
    w_values =  np.asarray(
            np.random.uniform(low=-bound, high=bound, size=filter_shape),
            dtype=d_type)
    b_values = np.zeros((filter_shape[0],), dtype=d_type)
    w.set_value(w_values), b.set_value(b_values)
    return

def set_weights_bias2(filter_shape, d_type, w, b):
    fan_in = filter_shape[1]
    fan_out = filter_shape[0]
     
    bound = np.sqrt(6. / (fan_in + fan_out))
    w_values =  np.asarray(
            np.random.uniform(low=-bound, high=bound, size=filter_shape),
            dtype=d_type)
    b_values = np.zeros((filter_shape[1],), dtype=d_type)
    w.set_value(w_values), b.set_value(b_values)
    return

def model(X, w1, b1, w2, b2,w3,b3,w4,b4):
    #pool dimensions for both the layers
    pool_dim = (2, 2)
    #First Convolutional Layer
    y1 = T.nnet.relu(conv2d(X, w1) + b1.dimshuffle('x', 0, 'x', 'x'))
    
    # First Pooling Layer
    o1 = pool.pool_2d(y1, pool_dim,ignore_border=True)
    
    
    
    
    # Second Convolutional Layer
    y2 = T.nnet.relu(conv2d(o1, w2) + b2.dimshuffle('x', 0, 'x', 'x'))
    # Second Pooling Layer
    o2 =pool.pool_2d(y2, pool_dim,ignore_border=True)
    
    

    # Flattening to 2 dimensions
    flat = T.flatten(o2, outdim=2)

    #Fully Connected Layer F3
    o3=T.nnet.relu(T.dot(flat,w3)+b3)

    # Softmax layer 
    pyx = T.nnet.softmax(T.dot(o3, w4) + b4)

    return y1,o1,y2,o2,o3, pyx

def shuffle_data (samples, labels):
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    samples, labels = samples[idx], labels[idx]
    return samples, labels

def sgd(cost, params, lr=0.05, decay=1e-4):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - (g + decay*p) * lr])
    return updates

def sgd_momentum(cost, params, lr=0.05, decay=0.0001, momentum=0.1):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        v = theano.shared(p.get_value()*0.)
        v_new = momentum*v - (g + decay*p) * lr 
        updates.append([p, p + v_new])
        updates.append([v, v_new])
    return updates

def RMSprop(cost, params, lr=0.001, decay=0.0001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * (g+ decay*p)))
    return updates


    
trX, teX, trY, teY = mnist(onehot=True)

trX = trX.reshape(-1, 1, 28, 28)
teX = teX.reshape(-1, 1, 28, 28)

trX, trY = trX[:12000], trY[:12000]
teX, teY = teX[:2000], teY[:2000]


X = T.tensor4('X')
Y = T.matrix('Y')


# no of feature maps in the first convolutional layer
num_filters_1 = 15
# no of feature maps in the second convolutional layer
num_filters_2 = 20

# weights and biases for the first convolutional layer
w1, b1 = init_weights_bias4((num_filters_1, 1, 9, 9), X.dtype)
#weights and biases for the second convolutional layer
w2, b2 = init_weights_bias4((num_filters_2, num_filters_1, 5, 5), X.dtype)
#weights and biases for the fully connected layer
w3, b3 = init_weights_bias2((num_filters_2*3*3, 100), X.dtype)
#weights and biases for the softmax layer
w4, b4 = init_weights_bias2((100, 10), X.dtype)



y1, o1,y2,o2,o3, py_x  = model(X, w1, b1, w2, b2,w3,b3,w4,b4)

# Final ouput by choosing the maximum probability
y_x = T.argmax(py_x, axis=1)

# Cost for back propogation using crossentropy function
cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))

# parameters to update
params = [w1, b1, w2, b2,w3,b3,w4,b4]



updates1 = sgd(cost, params)
updates2 = sgd_momentum(cost, params)
updates3 = RMSprop(cost, params)

train1 = theano.function(inputs=[X, Y], outputs=cost, updates=updates1, allow_input_downcast=True)
train2 = theano.function(inputs=[X, Y], outputs=cost, updates=updates2, allow_input_downcast=True)
train3 = theano.function(inputs=[X, Y], outputs=cost, updates=updates3, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)
test = theano.function(inputs = [X], outputs=[y1, o1], allow_input_downcast=True)



a1=[]
a2=[]
a3=[]
train_cost1=np.zeros(noIters)
train_cost2=np.zeros(noIters)
train_cost3=np.zeros(noIters)

print('sgd ..')

for i in range(noIters):
    trX, trY = shuffle_data (trX, trY)
    batch_cost=0
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        cost = train1(trX[start:end], trY[start:end])
        batch_cost+=cost
    a1.append(np.mean(np.argmax(teY, axis=1) == predict(teX)))
    train_cost1[i]=batch_cost/(noIters/batch_size)
    

#pylab.plot(range(noIters), a, label='sgd')

print('sgd with momentum ..')
# weights and biases for the first convolutional layer
set_weights_bias4((num_filters_1, 1, 9, 9), X.dtype,w1,b1)
#weights and biases for the second convolutional layer
set_weights_bias4((num_filters_2, num_filters_1, 5, 5), X.dtype,w2,b2)
#weights and biases for the fully connected layer
set_weights_bias2((num_filters_2*3*3, 100), X.dtype,w3,b3)
#weights and biases for the softmax layer
set_weights_bias2((100, 10), X.dtype,w4,b4)



for i in range(noIters):
    trX, trY = shuffle_data (trX, trY)
    batch_cost=0
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        cost = train2(trX[start:end], trY[start:end])
        batch_cost+=cost
    a2.append(np.mean(np.argmax(teY, axis=1) == predict(teX)))
    train_cost2[i]=batch_cost/(noIters/batch_size)
    

#pylab.plot(range(noIters), a, label='sgd with momentum')

print('RMSprop ..')
# weights and biases for the first convolutional layer
set_weights_bias4((num_filters_1, 1, 9, 9), X.dtype,w1,b1)
#weights and biases for the second convolutional layer
set_weights_bias4((num_filters_2, num_filters_1, 5, 5), X.dtype,w2,b2)
#weights and biases for the fully connected layer
set_weights_bias2((num_filters_2*3*3, 100), X.dtype,w3,b3)
#weights and biases for the softmax layer
set_weights_bias2((100, 10), X.dtype,w4,b4)


for i in range(noIters):
    trX, trY = shuffle_data (trX, trY)
    batch_cost=0
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        cost = train3(trX[start:end], trY[start:end])
        batch_cost+=cost
    a3.append(np.mean(np.argmax(teY, axis=1) == predict(teX)))
    train_cost3[i]=batch_cost/(noIters/batch_size)
    

#pylab.plot(range(noIters), a, label='RMSProp')

plt.figure(1)
plt.plot(range(noIters),a1,label="sgd")
plt.plot(range(noIters),a2,label="mom_sgd")
plt.plot(range(noIters),a3,label="RMSprop")
pylab.xlabel('epochs')
pylab.ylabel('test accuracy')
pylab.legend(loc='lower right')
pylab.savefig('combined_test_accuracy')


plt.figure(2)
plt.plot(range(noIters),train_cost1,label="sgd")
plt.plot(range(noIters),train_cost2,label="mom_sgd")
plt.plot(range(noIters),train_cost3,label="RMSprop")
pylab.xlabel('epochs')
pylab.ylabel('train cost')
pylab.legend(loc='upper right')
pylab.savefig('combined_training_accuracy')


pylab.show()
