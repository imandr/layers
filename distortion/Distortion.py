from keras import backend as K
from keras.layers import Layer

class Distortion(Layer):
    
    def __init__(self, output_dim, **kwargs):
        self.dim = output_dim
        super(Distortion, self).__init__(**kwargs)
        
    def initializer(self, shape, dtype=None):
        return (K.random_uniform(shape, dtype=dtype)*2-1)*0.2
        

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        
        assert len(input_shape) == 2 and input_shape[1] == self.dim
        
        self.corners = self.add_weight(name='kernel', 
                                      shape=(self.dim, self.dim),
                                      initializer=self.initializer,
                                      trainable=True)
        super(Distortion, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        #
        # x[i] -> x[i] + sum(k, Corner(k, i) * x[k] * prod(j, j!=k, (1-x[j])))
        #
        # x: [mb, n]
        # w = [mb, n, 1] dot [1, n]
        # R: [mb, n, n]
        R = K.dot(1.0-x[...,None], K.ones((1, self.dim)))   # [mb, n, 1], [1, n] -> [mb, n, n]
        diagonal = K.eye(self.dim)[None,...]
        R = R * (1.0 - diagonal) + diagonal             # R[mb, i, j] = { 1 - x[mb, i] if i != j, else 1 }
        Q = x * K.prod(R, axis=1)
        return x + K.dot(Q, self.corners)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dim)