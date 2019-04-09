from keras import backend as K
from keras.layers import Layer

class Bilinear(Layer):
    
    def __init__(self, **kwargs):
        self.dim = None
        super(Bilinear, self).__init__(**kwargs)
        
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        
        self.dim = input_shape[-1]
        
        self.mix = self.add_weight(name='kernel', 
                                      shape=(self.dim, self.dim),
                                      initializer="uniform",
                                      trainable=True)
        self.bias = self.add_weight(name='kernel', 
                                      shape=(self.dim,),
                                      initializer="uniform",
                                      trainable=True)
        super(Bilinear, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        shifted = K.concatenate([x[...,1:], x[...,:1]], axis=-1)
        bilinear = x*shifted
        return K.dot(bilinear, self.mix) + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape