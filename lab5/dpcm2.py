import numpy as np

from lab5.dpcm import Compress, Decompress


class Compress2(Compress):
    def __init__(self, filename):
        super(Compress2, self).__init__(filename)

    @staticmethod
    def mapping1(x):
        c = min(8, round(np.log(abs(x) + 1)))
        return min(15, 8 + np.sign(x) * c)

    @staticmethod
    def mapping2(x):
        sig = 1 if x >= 8 else -1
        c = (x - 8) * sig
        return (np.exp(c) - 1) * sig

    def compress(self):
        dif = []
        tmp = self._sig[0]
        for i, x in enumerate(self._sig):
            if i == 0:
                continue
            dif.append(self.mapping1(x - tmp))
            tmp = min(32767, max(tmp + self.mapping2(dif[-1]), -32768))

        dif = np.array(dif, dtype=np.uint8)
        with open('compressed2/{}.dpc'.format(self._filename), 'wb') as f:
            f.write(self._sig[0])
            for i in range(len(dif)):
                if i % 2 == 0:
                    continue
                f.write(np.uint8((dif[i-1] << 4) + dif[i]))


class Decompress2(Decompress):
    def __init__(self, filename):
        super(Decompress2, self).__init__(filename, 1)

    @staticmethod
    def mapping(x):
        sig = 1 if x >= 8 else -1
        c = (x - 8) * sig
        return (np.exp(c) - 1) * sig

    def decompress(self):
        sig = [self._head]
        for x in self._dif:
            sig.append(min(32767, max(sig[-1] + self.mapping(x), -32768)))

        return np.array(sig, dtype=np.int16)
