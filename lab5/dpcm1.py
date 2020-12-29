import numpy as np

from lab5.dpcm import Compress, Decompress


class Compress1(Compress):
    """ subclass of compress, implement the 8-bit compress with factor quantization

    Args:
        base (int): factor for quantization, default: 100
    """
    def __init__(self, filename, base=100):
        super(Compress1, self).__init__(filename)
        self._base = base

    def mapping(self, x):
        """ evaluate dif value into the quantized level

        :param x: dif value
        :return: quantized level
        """
        x = round(x / self._base)
        return min(127, max(x, -128))

    def compress(self):
        # compress the original signal
        dif = []
        tmp = self._sig[0]
        for i, x in enumerate(self._sig):
            if i == 0:
                continue
            dif.append(self.mapping(x - tmp))
            tmp = min(32767, max(tmp + dif[-1] * self._base, -32768))

        # write the compressed signal into the disk
        dif = np.array(dif, dtype=np.int8)
        with open('compressed1/{}_8bit.dpc'.format(self._filename), 'wb') as f:
            f.write(np.uint8(self._sig[0] >> 8))
            f.write(np.uint8(self._sig[0] & 0x00ff))
            for x in dif:
                f.write(x)


class Decompress1(Decompress):
    """ subclass of decompress, implement the 8-bit compress with factor quantization

    Args:
        base (int): factor for quantization, default: 100
    """
    def __init__(self, filename, base=100):
        self._base = base
        super(Decompress1, self).__init__(filename, 2)

    def decompress(self):
        sig = [self._head]
        for x in self._dif:
            sig.append(min(32767, max(sig[-1] + x * self._base, -32768)))

        return np.array(sig, dtype=np.int16)
