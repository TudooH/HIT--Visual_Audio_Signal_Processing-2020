import numpy as np


with open('compressed1/1.dpc', 'rb') as f:
    m = f.read()
    print(m[0], m[0] << 8)
    print(m[1])
    print(np.int16((m[0] << 8) + m[1]))

