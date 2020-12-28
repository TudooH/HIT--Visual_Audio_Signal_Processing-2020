import numpy as np
import wave

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

        # return self._sig[0], dif


class Decompress1(Decompress):
    def __init__(self, filename, base):
        self._base = base
        super(Decompress1, self).__init__(filename, 1)

    def decompress(self):
        sig = [self._head]
        for x in self._dif:
            sig.append(min(32767, max(sig[-1] + x * self._base, -32768)))

        return np.array(sig, dtype=np.int16)

    def write_file(self, filename):
        with wave.open('decompressed1/{}.pcm'.format(filename), 'wb') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.writeframes(self._sig)


for i in range(10):
    co = Compress1(str(i+1), 100)
    co.compress()
    de = Decompress1('compressed1/{}.dpc'.format(str(i+1)), 100)
    de.write_file(str(i+1))
