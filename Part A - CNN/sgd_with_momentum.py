from load import mnist
import numpy as np
import pylab 

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

def sgd_momentum(cost, params, lr=0.05, decay=1e-4, momentum=0.1):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        v = theano.shared(p.get_value()*0.)
        v_new = momentum*v - (g + decay*p) * lr 
        updates.append([p, p + v_new])
        updates.append([v, v_new])
    return updates

def shuffle_data (samples, labels):
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    samples, labels = samples[idx], labels[idx]
    return samples, labels


# Preprocessing the train set and the test set    
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

updates = sgd_momentum(cost, params)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)
test = theano.function(inputs = [X], outputs=[y1, o1,y2,o2], allow_input_downcast=True)

#  array for test accuracy
a = []
train_cost=np.zeros(noIters)


for i in range(noIters):
    print(i)
    # Shuffle data for each iteration for faster convergence 
    trX, trY = shuffle_data (trX, trY)
    teX, teY = shuffle_data (teX, teY)
    batch_cost=0

    # mini batch gradient with batch size 128
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        cost= train(trX[start:end], trY[start:end])
        batch_cost+=cost
    a.append(np.mean(np.argmax(teY, axis=1) == predict(teX)))
    train_cost[i]=batch_cost/(noIters/batch_size)
    print(a[i])
    


# Plotting training error
pylab.figure()
pylab.plot(range(noIters), train_cost)
pylab.title("sgd")
pylab.xlabel('epochs')
pylab.ylabel('train error')
pylab.savefig('mom_train_error.png')



# Plotting test accuracy
pylab.figure()
pylab.plot(range(noIters), a)
pylab.xlabel('epochs')
pylab.ylabel('test accuracy')
pylab.savefig('mom_test_accuracy_sgd_momentum.png')

'''
w = w1.get_value()
pylab.figure()
pylab.gray()
for i in range(25):
    pylab.subplot(5, 5, i+1); pylab.axis('off'); pylab.imshow(w[i,:,:,:].reshape(9,9))
#pylab.title('filters learned')
pylab.savefig('mom_figure_2a_2.png')
'''


# images at convolutional and pooling layers for 2 samples
for sample in range(2):
    ind = np.random.randint(low=0, high=2000)
    convolved1, pooled1,convolved2,pooled2 = test(teX[ind:ind+1,:])

    pylab.figure()
    pylab.gray()
    pylab.axis('off'); pylab.imshow(teX[ind,:].reshape(28,28))
    #pylab.title('input image')
    pylab.savefig('mom'+str(sample)+'_question1_input.png')

    pylab.figure()
    pylab.gray()
    for i in range(num_filters_1):
        pylab.subplot(5, 3, i+1); pylab.axis('off'); pylab.imshow(convolved1[0,i,:].reshape(20,20))
    #   pylab.title('convolved feature maps')
    pylab.savefig('mom'+str(sample)+'_question1_convolution1.png')

    pylab.figure()
    pylab.gray()
    for i in range(num_filters_1):
        pylab.subplot(5, 3, i+1); pylab.axis('off'); pylab.imshow(pooled1[0,i,:].reshape(10,10))
    #   pylab.title('pooled feature maps')
    pylab.savefig('mom'+str(sample)+'_question1_pooling1.png')

    pylab.figure()
    pylab.gray()
    for i in range(num_filters_2):
        pylab.subplot(5, 4, i+1); pylab.axis('off'); pylab.imshow(convolved2[0,i,:].reshape(6,6))
    #   pylab.title('convolved feature maps')
    pylab.savefig('mom'+str(sample)+'_question1_convolution2.png')

    pylab.figure()
    pylab.gray()
    for i in range(num_filters_2):
        pylab.subplot(5, 4, i+1); pylab.axis('off'); pylab.imshow(pooled2[0,i,:].reshape(3,3))
    #   pylab.title('pooled feature maps')
    pylab.savefig('mom'+str(sample)+'_question1_pooling2.png')


pylab.show()


