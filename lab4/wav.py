from scipy.io import wavfile
import numpy as np
import wave


def sgn(x):
    return 1 if x >= 0 else -1


class Wav:
    def __init__(self, filename):
        self._filename = filename
        _, sig = wavfile.read('data/{}.wav'.format(filename))
        self._sig = np.array(sig, dtype=np.int64)
        self._num = int(len(self._sig) / 256)
        if len(self._sig) % 256 != 0:
            self._num += 1

    def get_energy(self):
        energy = []
        for i in range(self._num - 1):
            energy.append(sum(np.power(self._sig[i*256: (i+1)*256], 2)))
        energy.append(sum(np.power(self._sig[(self._num-1)*256:], 2)))

        return energy

    def get_zeros(self):
        zeros = []
        p, tmp = 0, 0
        for i, x in enumerate(self._sig):
            if (i + 1) % 256 == 0 or i == len(self._sig)-1:
                zeros.append(tmp / (i-p) / 2)
                p, tmp = i+1, 0
                continue
            tmp += abs(sgn(self._sig[i]) - sgn(self._sig[i+1]))

        return zeros

    def detect(self):
        energy = self.get_energy()
        zeros = self.get_zeros()

        high = np.mean(energy) / 2
        low = high / 4
        low_zeros = np.mean(zeros) * 1.5

        p = 0
        voiced = []
        while p < len(energy):
            while p < len(energy) and energy[p] < low:
                p += 1
            tmp = p
            while tmp < len(energy) and energy[tmp] >= low:
                tmp += 1

            if tmp - p > 5 and [x for x in energy[p: tmp] if x >= high]:
                voiced.append([p, tmp])
            p = tmp + 1

        flag = [False] * self._num
        for i, v in enumerate(voiced):
            p1 = 0 if i == 0 else voiced[i-1][1]
            p2 = v[0] - 1
            while p2 >= p1 and zeros[p2] <= low_zeros:
                p2 -= 1
            voiced[i][0] = p2 + 1

            for j in range(voiced[i][0], voiced[i][1]):
                flag[j] = True

        new_sig = []
        for i in range(self._num):
            time = 256 if i != self._num-1 else len(self._sig) % 256
            if flag[i]:
                for j in range(time):
                    new_sig.append(self._sig[i * 256 + j])
            else:
                for j in range(time):
                    new_sig.append(0)
        new_sig = np.array(new_sig, dtype=np.int16)
        with wave.open('pcm/{}.pcm'.format(self._filename), 'wb') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.writeframes(new_sig.tobytes())


def write_file(filename, a):
    with open(filename, 'w') as f:
        for x in a:
            f.write(str(x)+'\n')


if __name__ == '__main__':
    for num in range(10):
        wav = Wav(str(num+1))

        en = wav.get_energy()
        write_file('en/{}_en.txt'.format(str(num+1)), en)

        zo = wav.get_zeros()
        write_file('zero/{}_zero.txt'.format(str(num + 1)), zo)

        wav.detect()
