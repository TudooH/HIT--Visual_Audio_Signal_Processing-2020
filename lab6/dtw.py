from struct import unpack
import numpy as np


def get_feature(filename):
    feature = []
    with open('mfc/{}.mfc'.format(filename), 'rb') as f:
        frames = unpack(">i", f.read(4))[0]
        f.read(4)
        byte = unpack(">h", f.read(2))[0]
        f.read(2)
        dim = byte // 4
        for m in range(frames):
            feature_frame = []
            for n in range(dim):
                feature_frame.append(unpack(">f", f.read(4))[0])
            feature.append(feature_frame)
    return np.array(feature)


def get_dis(f1, f2):
    dis = []
    for x in f1:
        tmp = []
        for y in f2:
            tmp.append(np.sqrt(np.sum(np.power(x-y, 2))))
        dis.append(tmp)
    return dis


def dtw(f1, f2):
    dis = get_dis(f1, f2)
    dp = np.zeros((len(dis)+1, len(dis[0])+1))
    for i in range(1, dp.shape[0]):
        for j in range(1, dp.shape[1]):
            weight = dis[i-1][j-1]
            dp[i][j] = min(dp[i-1][j-1] + 2*weight, dp[i-1][j] + weight, dp[i][j-1] + weight)
    return dp[dp.shape[0]-1][dp.shape[1]-1] / (dp.shape[0] + dp.shape[1] - 2)
