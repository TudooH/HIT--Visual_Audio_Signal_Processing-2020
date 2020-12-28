import numpy as np

from lab5.dpcm import Compress, Decompress


class Compress1(Compress):
    def __init__(self, filename, base):
        super(Compress1, self).__init__(filename)
        self._base = base

    def mapping(self, x):
        x = round(x / self._base)
        return min(127, max(x, -128))

    def compress(self):
        dif = []
        tmp = self._sig[0]
        for i, x in enumerate(self._sig):
            if i == 0:
                continue
            dif.append(self.mapping(x - tmp))
            tmp = min(32767, max(tmp + dif[-1] * self._base, -32768))

        dif = np.array(dif, dtype=np.int8)

        with open('compressed1/{}.dpc'.format(self._filename), 'wb') as f:
            f.write(self._sig[0])
            for x in dif:
                f.write(x)


class Decompress1(Decompress):
    def __init__(self, filename, base):
        self._base = base
        super(Decompress1, self).__init__(filename, 2)

    def decompress(self):
        sig = [self._head]
        for x in self._dif:
            sig.append(min(32767, max(sig[-1] + x * self._base, -32768)))

        return np.array(sig, dtype=np.int16)
