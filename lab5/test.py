from lab5.dpcm1 import Compress1, Decompress1
from lab5.dpcm2 import Compress2, Decompress2


for i in range(10):
    co = Compress1(str(i+1), 100)
    co.compress()
    de = Decompress1('compressed1/{}'.format(str(i+1)), 100)

for i in range(10):
    co = Compress2(str(i+1))
    co.compress()
    de = Decompress2('compressed2/{}'.format(str(i+1)))