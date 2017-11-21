from load import mnist
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams 
import numpy as np
import pylab
import time

no_of_inputs = 784
no_of_hidden_layer_1_neurons = 900
no_of_hidden_layer_2_neurons = 625
no_of_hidden_layer_3_neurons = 400
corruption_level = 0.1
# We experiment with a larger number of epochs
training_epochs = 100
learning_rate = 0.1
batch_size = 128


def init_weights(n_visible, n_hidden):
    """ Get the initial values for the weights """

    initial_W = np.asarray(
        np.random.uniform(
            low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
            high=4 * np.sqrt(6. / (n_hidden + n_visible)),
            size=(n_visible, n_hidden)),
        dtype=theano.config.floatX)
    return theano.shared(value=initial_W, name='W', borrow=True)


def init_bias(no_of_neurons=1):
    """ Set initial values for the bias """

    return theano.shared(np.zeros(no_of_neurons), theano.config.floatX)

# Read the training and testing data
trX, teX, trY, teY = mnist()

x = T.fmatrix('x')  
d = T.fmatrix('d')

start_time = time.time()

rng = np.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))

# Initialize weights and biases for the first hidden layer
W1 = init_weights(no_of_inputs, no_of_hidden_layer_1_neurons)
b1 = init_bias(no_of_hidden_layer_1_neurons)
b1_prime = init_bias(no_of_inputs)
W1_prime = W1.transpose() 

# Initialize weights and biases for the second hidden layer
W2 = init_weights(no_of_hidden_layer_1_neurons, no_of_hidden_layer_2_neurons)
b2 = init_bias(no_of_hidden_layer_2_neurons)
W2_prime = W2.transpose()
b2_prime = init_bias(no_of_hidden_layer_1_neurons)

# Initialize weights and biases for the third hidden layer
W3 = init_weights(no_of_hidden_layer_2_neurons, no_of_hidden_layer_3_neurons)
b3 = init_bias(no_of_hidden_layer_3_neurons)
W3_prime = W3.transpose()
b3_prime = init_bias(no_of_hidden_layer_2_neurons)

# Corrupt input data using multiplicative noise with binomial distribution 
tilde_x = theano_rng.binomial(size=x.shape, n=1, p=1-corruption_level, dtype=theano.config.floatX)*x

y1 = T.nnet.sigmoid(T.dot(tilde_x, W1) + b1)
z1 = T.nnet.sigmoid(T.dot(y1, W1_prime) + b1_prime)
cost1 = - T.mean(T.sum(x * T.log(z1) + (1 - x) * T.log(1 - z1), axis=1))

params1 = [W1, b1, b1_prime]
grads1 = T.grad(cost1, params1)
updates1 = [(param1, param1 - learning_rate * grad1) for param1, grad1 in zip(params1, grads1)]
train_da1 = theano.function(inputs=[x], outputs = cost1, updates = updates1, allow_input_downcast = True)

y2 = T.nnet.sigmoid(T.dot(y1, W2) + b2)
z2 = T.nnet.sigmoid(T.dot(y2, W2_prime) + b2_prime)
cost2 = - T.mean(T.sum(y1 * T.log(z2) + (1 - y1) * T.log(1 - z2), axis=1))

params2 = [W2, b2, b2_prime]
grads2 = T.grad(cost2, params2)
updates2 = [(param2, param2 - learning_rate * grad2) for param2, grad2 in zip(params2, grads2)]
train_da2 = theano.function(inputs=[x], outputs = cost2, updates = updates2, allow_input_downcast = True)

# Build a stacked denoising autoencoder
y3 = T.nnet.sigmoid(T.dot(y2, W3) + b3)
z3 = T.nnet.sigmoid(T.dot(y3, W3_prime) + b3_prime)
cost3 = - T.mean(T.sum(y2 * T.log(z3) + (1 - y2) * T.log(1 - z3), axis=1))
z4 = T.nnet.sigmoid(T.dot(z3, W2_prime) + b2_prime)
z5 = T.nnet.sigmoid(T.dot(z4, W1_prime) + b1_prime)

params3 = [W3, b3, b3_prime]
grads3 = T.grad(cost3, params3)
updates3 = [(param3, param3 - learning_rate * grad3) for param3, grad3 in zip(params3, grads3)]
train_da3 = theano.function(inputs=[x], outputs=cost3, updates=updates3, allow_input_downcast=True)
test_da3 = theano.function(inputs=[x], outputs=[y1, y2, y3, z5], allow_input_downcast=True)

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
pylab.savefig('2b1_1_learning_curve_layer_1.png')

w1 = W1.get_value()
pylab.figure()
pylab.gray()

for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(w1[:,i].reshape(28,28))
pylab.savefig('2b1_1_weights_layer_1.png')

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
pylab.savefig('2b1_2_learning_curve_layer_2.png')

w2 = W2.get_value()
pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(w2[:,i].reshape(30,30))
pylab.savefig('2b1_2_weights_layer_2.png')

print('Training Denoising Autoencoder Layer 3')
d = []
for epoch in range(training_epochs):
        c = []
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
            c.append(train_da3(trX[start:end]))
        d.append(np.mean(c, dtype='float64'))
        print(d[epoch])
        
end_time = time.time()
print("Total time taken: {}".format(end_time - start_time))

pylab.figure()
pylab.plot(range(training_epochs), d)
pylab.xlabel('No. of Iterations')
pylab.ylabel('Cross Entropy')
pylab.title('Learning Curve for Layer 3')
pylab.savefig('2b1_2_learning_curve_layer_3.png')

w3 = W3.get_value()
pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(w3[:,i].reshape(25,25))
pylab.savefig('2b1_3_weights_layer_3.png')

# Plot the input images for 100 samples
pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(teX[i,:].reshape(28,28))
pylab.savefig('2b1_input_images.png')
pylab.show()

# We get the activations of hidden layers 1, 2 and 3 
# as well as the output of the autoencoder
hl1_a, hl2_a, hl3_a, z = test_da3(teX)

# Plot the reconstructed images for 100 samples
pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(z[i,:].reshape(28,28))
pylab.savefig('2b1_reconstructed_images.png')
pylab.show()

# Plot the activations for hidden layer 1 (100 samples)
pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(hl1_a[i,:].reshape(30,30))
pylab.savefig('2b1_hidden_layer_1_activations.png')
pylab.show()

# Plot the activations for hidden layer 2 (100 samples)
pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(hl2_a[i,:].reshape(25,25))
pylab.savefig('2b1_hidden_layer_2_activations.png')
pylab.show()

# Plot the activations for hidden layer 3 (100 samples)
pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(hl3_a[i,:].reshape(20,20))
pylab.savefig('2b1_hidden_layer_3_activations.png')
pylab.show()
