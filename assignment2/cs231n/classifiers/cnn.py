from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################

        F = num_filters
        C, H, W = input_dim

        self.conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        self.pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        H_ = 1 + (H + 2 * self.conv_param['pad'] - filter_size) // self.conv_param['stride']
        W_ = 1 + (W + 2 * self.conv_param['pad'] - filter_size) // self.conv_param['stride']
        self.Hp = (H_ - self.pool_param['pool_height']) // self.pool_param['stride'] + 1
        self.Wp = (W_ - self.pool_param['pool_width']) // self.pool_param['stride'] + 1

        self.params['b1'] = np.zeros(F)
        self.params['W1'] = np.random.randn(F, C, filter_size, filter_size) * weight_scale
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.randn(F * self.Hp * self.Wp, hidden_dim) * weight_scale
        self.params['b3'] = np.zeros(num_classes)
        self.params['W3'] = np.random.randn(hidden_dim, num_classes) * weight_scale
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        N, _, _, _ = X.shape
        F, _, _, _ = W1.shape

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        # conv_h, conv_cache = conv_forward_fast(x, W1, b1, self.conv_param)
        # relu_h, relu_cache = relu_forward(conv_h)
        # pool_h, pool_cache = max_pool_forward_fast(relu_h, self.pool_param)

        h, conv_relu_pool_cache = conv_relu_pool_forward(X, W1, b1, self.conv_param, self.pool_param)
        af_h, af_cache = affine_relu_forward(h, W2, b2)
        scores, output_cache = affine_forward(af_h, W3, b3)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(W3 ** 2))

        dx1, dW3, db3 = affine_backward(dscores, output_cache)
        dW3 += self.reg * W3

        dx2, dW2, db2 = affine_relu_backward(dx1, af_cache)
        dW2 += self.reg * W2

        dx2 = dx2.reshape(N, F, self.Hp, self.Wp)
        dx, dW1, db1 = conv_relu_pool_backward(dx2, conv_relu_pool_cache)
        dW1 += self.reg * W1
        grads = dict(W1=dW1, b1=db1, W2=dW2, b2=db2, W3=dW3, b3=db3)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


# def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
#     conv_h, conv_cache = conv_forward_fast(x, w, b, conv_param)
#     relu_h, relu_cache = relu_forward(conv_h)
#     pool_h, pool_cache = max_pool_forward_fast(relu_h, pool_param)
#     cache = (conv_cache, relu_cache, pool_cache)
#     return pool_h, cache
#
#
# def conv_relu_pool_backward(dout, cache):
#     conv_cache, relu_cache, pool_cache = cache
#     # print(pool_cache)
#     dx_pool = max_pool_backward_fast(dout, pool_cache)
#     dx_relu = relu_backward(dx_pool, relu_cache)
#     return conv_backward_fast(dx_relu, conv_cache)
