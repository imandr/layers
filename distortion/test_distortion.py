import math, random, numpy as np
from Distortion import Distortion
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam

Thickness = 0.01
Radius = 2.0
Position = 3.0

def generate_point(ring):
    phi = random.random()*2*math.pi
    r = random.random()*Thickness
    u, v = r*math.sin(phi), r*math.cos(phi) + Radius

    theta = random.random()*2*math.pi
    w = u
    u, v, w = v*math.cos(theta), v*math.sin(theta), u
    
    if ring == 0:
        x, y, z = u, v, w
    elif ring == 1:
        x, y, z = u + Position, w, v
    elif ring == 2:
        x, y, z = u - Position, w, v
    return np.array([x,y,z])     
    
def generate(mbsize):
    x = []
    y = []
    for _ in xrange(mbsize):
        ring = random.randint(0,2)
        point = generate_point(ring)
        x.append(point)
        r = np.zeros((3,), dtype=np.float)
        r[ring] = 1.0
        y.append(r)
    return np.array(x), np.array(y)
    
def generator(mbsize):
    while True:
        yield generate(mbsize)
    
def build_model():
    
    inp = Input((3,))
    l1 = Dense(4)(inp)
    l2 = Distortion(4)(l1)
    l3 = Dense(3, activation="softmax")(l2)
    model = Model(inputs=[inp], outputs = [l3])
    model.compile(Adam(lr=0.001), loss="categorical_crossentropy")
    return model




if __name__ == '__main__':
    x, y = generate(5)
    for point, ring in zip(x, y):
        print point, ring
        if ring[0]:
            r = math.sqrt(point[0]**2+point[1]**2)
            print r, point[2]
        if ring[1]:
            r = math.sqrt((point[0]-Position)**2+point[2]**2)
            print r, point[1]
        if ring[2]:
            r = math.sqrt((point[0]+Position)**2+point[2]**2)
            print r, point[1]
            
        
    