from load import mnist
import numpy as np
import pylab
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import time

corruption_level=0.1
# We experiment with a larger number of epochs
training_epochs = 100
learning_rate = 0.1
batch_size = 128
beta = 0.5
rho = 0.05

def init_weights(n_visible, n_hidden):
    initial_W = np.asarray(
        np.random.uniform(
            low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
            high=4 * np.sqrt(6. / (n_hidden + n_visible)),
            size=(n_visible, n_hidden)),
        dtype=theano.config.floatX)
    return theano.shared(value=initial_W, name='W', borrow=True)

	
def init_bias(n):
    return theano.shared(value=np.zeros(n,dtype=theano.config.floatX),borrow=True)


def sgd_momentum(cost, params, lr=0.1, decay=0.0001, momentum=0.1):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        v = theano.shared(p.get_value())
        v_new = momentum*v - (g + decay*p) * lr 
        updates.append([p, p + v_new])
        updates.append([v, v_new])
        return updates

trX, teX, trY, teY = mnist()

x = T.fmatrix('x')  
d = T.fmatrix('d')

rng = np.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))

W1 = init_weights(28*28, 900)
b1 = init_bias(900)
b1_prime = init_bias(28*28)
W1_prime = W1.transpose() 
W2 = init_weights(900, 625)
b2 = init_bias(625)
W2_prime = W2.transpose()
b2_prime = init_bias(900)
W3 = init_weights(625, 400)
b3 = init_bias(400)
W3_prime = W3.transpose()
b3_prime = init_bias(625)
W4 = init_weights(400, 10)
b4 = init_bias(10)

start_time = time.time()
tilde_x = theano_rng.binomial(size=x.shape, n=1, p=1 - corruption_level,
                              dtype=theano.config.floatX)*x
y1 = T.nnet.sigmoid(T.dot(tilde_x, W1) + b1)
z1 = T.nnet.sigmoid(T.dot(y1, W1_prime) + b1_prime)
cost1 = - T.mean(T.sum(x * T.log(z1) + (1 - x) * T.log(1 - z1), axis=1)) \
                + beta*T.shape(y1)[1]*(rho*T.log(rho) + (1-rho)*T.log(1-rho)) \
                - beta*rho*T.sum(T.log(T.mean(y1, axis=0)+1e-6)) \
                - beta*(1-rho)*T.sum(T.log(1-T.mean(y1, axis=0)+1e-6))


params1 = [W1, b1, b1_prime]
grads1 = T.grad(cost1, params1)
updates1 = sgd_momentum(cost1, params1)
train_da1 = theano.function(inputs=[x], outputs = cost1, updates = updates1, allow_input_downcast = True)

y2 = T.nnet.sigmoid(T.dot(y1, W2)+b2)
z2 = T.nnet.sigmoid(T.dot(y2, W2_prime) + b2_prime)
cost2 = - T.mean(T.sum(y1 * T.log(z2) + (1 - y1) * T.log(1 - z2), axis=1)) \
                + beta*T.shape(y2)[1]*(rho*T.log(rho) + (1-rho)*T.log(1-rho)) \
                - beta*rho*T.sum(T.log(T.mean(y2, axis=0)+1e-6)) \
                - beta*(1-rho)*T.sum(T.log(1-T.mean(y2, axis=0)+1e-6))



params2 = [W2, b2, b2_prime]
grads2 = T.grad(cost2, params2)
updates2 = sgd_momentum(cost2, params2)
train_da2 = theano.function(inputs=[x], outputs = cost2, updates = updates2, allow_input_downcast = True)

y3 = T.nnet.sigmoid(T.dot(y2, W3)+b3)
z3 = T.nnet.sigmoid(T.dot(y3, W3_prime) + b3_prime)
z4 = T.nnet.sigmoid(T.dot(z3, W2_prime)+ b2_prime)
z5 = T.nnet.sigmoid(T.dot(z4, W1_prime)+ b1_prime)

cost3 = - T.mean(T.sum(y2 * T.log(z3) + (1 - y2) * T.log(1 - z3), axis=1)) \
                + beta*T.shape(y3)[1]*(rho*T.log(rho) + (1-rho)*T.log(1-rho)) \
                - beta*rho*T.sum(T.log(T.mean(y3, axis=0)+1e-6)) \
                - beta*(1-rho)*T.sum(T.log(1-T.mean(y3, axis=0)+1e-6))

params3 = [W3, b3, b3_prime]
grads3 = T.grad(cost3, params3)
updates3 = sgd_momentum(cost3, params3)
train_da3 = theano.function(inputs=[x], outputs = cost3, updates = updates3, allow_input_downcast = True)
test_da3 = theano.function(inputs=[x], outputs = z5, allow_input_downcast=True)

p_y4 = T.nnet.softmax(T.dot(y3, W4)+b4)
y4 = T.argmax(p_y4, axis=1)
cost4 = T.mean(T.nnet.categorical_crossentropy(p_y4, d))

params4 = [W1, b1, W2, b2, W3, b3, W4, b4]
grads4 = T.grad(cost4, params4)
updates4 = sgd_momentum(cost4, params4)
train_ffn = theano.function(inputs=[x, d], outputs = cost4, updates = updates4, allow_input_downcast = True)
test_ffn = theano.function(inputs=[x], outputs = [y1, y2, y3, z5, y4], allow_input_downcast=True)


print('Training Denoising Autoencoder Layer 1')
d = []
for epoch in range(training_epochs):
        c = []
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
            c.append(train_da1(trX[start:end]))
        d.append(np.mean(c, dtype='float64'))
        print(d[epoch])

pylab.figure()
pylab.plot(range(training_epochs), d)
pylab.xlabel('No. of Iterations')
pylab.ylabel('Cross Entropy')
pylab.title('Learning Curve for Layer 1')
pylab.savefig('2b3_1_learning_curve_layer_1.png')

w1 = W1.get_value()
pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(w1[:,i].reshape(28,28))
pylab.savefig('2b3_1_weights_layer_1.png')

print('Training Denoising Autoencoder Layer 2')
d = []
for epoch in range(training_epochs):
        c = []
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
            c.append(train_da2(trX[start:end]))
        d.append(np.mean(c, dtype='float64'))
        print(d[epoch])

pylab.figure()
pylab.plot(range(training_epochs), d)
pylab.xlabel('No. of Iterations')
pylab.ylabel('Cross Entropy')
pylab.title('Learning Curve for Layer 2')
pylab.savefig('2b3_2_learning_curve_layer_2.png')

w2 = W2.get_value()
pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(w2[:,i].reshape(30,30))
pylab.savefig('2b3_2_weights_layer_2.png')

print('Training Denoising Autoencoder Layer 3')
d = []
for epoch in range(training_epochs):
        c = []
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
            c.append(train_da3(trX[start:end]))
        d.append(np.mean(c, dtype='float64'))
        print(d[epoch])

pylab.figure()
pylab.plot(range(training_epochs), d)
pylab.xlabel('No. of Iterations')
pylab.ylabel('Cross Entropy')
pylab.title('Learning Curve for Layer 3')
pylab.savefig('2b3_2_learning_curve_layer_3.png')

w3 = W3.get_value()
pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(w3[:,i].reshape(25,25))
pylab.savefig('2b3_3_weights_layer_3.png')


print('Training the feedforward network')
d, a = [], []
for epoch in range(training_epochs):
        c = []
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
            c.append(train_ffn(trX[start:end], trY[start:end]))
        d.append(np.mean(c, dtype='float64'))
        a.append(np.mean(np.argmax(teY, axis=1) == test_ffn(teX)[-1]))
        print(a[epoch])

end_time = time.time()
print ("Time Taken =", end_time - start_time )

pylab.figure()
pylab.plot(range(training_epochs), d)
pylab.xlabel('No. of Iterations')
pylab.ylabel('Categorical Cross Entropy')
pylab.title('Training Error vs No. of Iterations')
pylab.savefig('2b3_training_error.png')

pylab.figure()
pylab.plot(range(training_epochs), a)
pylab.xlabel('No. of Iterations')
pylab.ylabel('Test Accuracy')
pylab.title('Test Accuracy vs No. of Iterations')
pylab.savefig('2b3_test_accuracy.png')
pylab.show()

# Plot the input images for 100 samples
pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(teX[i,:].reshape(28,28))
pylab.savefig('2b3_input_images.png')
pylab.show()

# We get the activations of hidden layers 1, 2 and 3 
# as well as the output of the autoencoder
hl1_a, hl2_a, hl3_a, z, out = test_ffn(teX)

# Plot the reconstructed images for 100 samples
pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(z[i,:].reshape(28,28))
pylab.savefig('2b3_reconstructed_images.png')
pylab.show()

# Plot the activations for hidden layer 1 (100 samples)
pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(hl1_a[i,:].reshape(30,30))
pylab.savefig('2b3_hidden_layer_1_activations.png')
pylab.show()

# Plot the activations for hidden layer 2 (100 samples)
pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(hl2_a[i,:].reshape(25,25))
pylab.savefig('2b3_hidden_layer_2_activations.png')
pylab.show()

# Plot the activations for hidden layer 3 (100 samples)
pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(hl3_a[i,:].reshape(20,20))
pylab.savefig('2b3_hidden_layer_3_activations.png')
pylab.show()



