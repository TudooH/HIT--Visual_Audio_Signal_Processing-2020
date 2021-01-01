from struct import unpack
import numpy as np


def get_feature(filename):
    """ get the feature from the mfc file

    :param filename: seq of the mfc filename, for example: 1_1
    :return: list, each item in the list is a list of 39 nums
    """
    feature = []
    with open('mfc/{}.mfc'.format(filename), 'rb') as f:
        frames = unpack(">i", f.read(4))[0]  # number of frame
        f.read(4)
        byte = unpack(">h", f.read(2))[0]  # number of byte, = dim * 4
        f.read(2)
        dim = byte // 4
        for m in range(frames):
            feature_frame = []
            for n in range(dim):
                feature_frame.append(unpack(">f", f.read(4))[0])
            feature.append(feature_frame)
    return np.array(feature)


def get_dis(f1, f2):
    """ get the distance between the items of f1 and f2 one by one

    :param f1: the first feature get from get_feature()
    :param f2: the second feature get from get_feature()
    :return: 2-dim list, for example: dis[i][j] represents distance between i in f1 and j in f2
    """
    dis = []
    for x in f1:
        tmp = []
        for y in f2:
            tmp.append(np.sqrt(np.sum(np.power(x-y, 2))))
        dis.append(tmp)
    return dis


def dtw(f1, f2):
    """ get the dtw distance between f1 and f2

    :param f1: the first feature get from get_feature()
    :param f2: the second feature get from get_feature()
    :return: the dtw distance between f1 and f2
    """
    dis = get_dis(f1, f2)  # get the distance of f1 and f2
    dp = np.zeros((len(dis)+1, len(dis[0])+1))  # initialize the dp list
    for i in range(1, dp.shape[0]):
        for j in range(1, dp.shape[1]):
            weight = dis[i-1][j-1]
            dp[i][j] = min(dp[i-1][j-1] + 2*weight, dp[i-1][j] + weight, dp[i][j-1] + weight)  # select the min value
    return dp[dp.shape[0]-1][dp.shape[1]-1] / (dp.shape[0] + dp.shape[1] - 2)
