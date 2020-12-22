from scipy.io import wavfile
import numpy as np
import wave


class Compress:
    def __init__(self, filename, base):
        self._filename = filename
        _, sig = wavfile.read('../wav_source/{}.wav'.format(filename))
        self._sig = np.array(sig, dtype=np.int16)
        self._base = base

    def mapping(self, x):
        x = round(x / self._base)
        return min(127, max(x, -128))

    def compress(self):
        dif = []
        for i, x in enumerate(self._sig):
            if i == 0:
                continue
            dif.append(self.mapping(x - self._sig[i-1]))

        dif = np.array(dif, dtype=np.int8)

        with open('compressed/{}.dpc'.format(self._filename), 'wb') as f:
            f.write(self._sig[0])
            for x in dif:
                f.write(x)

        return self._sig[0], dif


class Decompress:
    def __init__(self, head, dif, base):
        sig = [head]
        for x in dif:
            sig.append(sig[-1] + x * base)

        self._sig = np.array(sig, dtype=np.int16)

    def write_file(self, filename):
        with wave.open('decompressed/{}.pcm'.format(filename), 'wb') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.writeframes(self._sig)


for i in range(10):
    co = Compress(str(i+1), 100)
    head, dif = co.compress()
    de = Decompress(head, dif, 100)
    de.write_file(str(i+1))
