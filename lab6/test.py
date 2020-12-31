from struct import unpack
import numpy as np
import csv


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


def dtw(dis):
    dp = np.zeros((len(dis)+1, len(dis[0])+1))
    for i in range(1, dp.shape[0]):
        for j in range(1, dp.shape[1]):
            weight = dis[i-1][j-1]
            dp[i][j] = min(dp[i-1][j-1] + 2*weight, dp[i-1][j] + weight, dp[i][j-1] + weight)
    return dp[dp.shape[0]-1][dp.shape[1]-1] / (dp.shape[0] + dp.shape[1] - 2)


# ans = np.zeros((50, 50))
# for i in range(50):
#     for j in range(i):
#         print(i, j)
#         x1 = i // 5 + 1
#         y1 = i % 5 + 1
#         x2 = j // 5 + 1
#         y2 = j % 5 + 1
#         ans[i][j] = dtw(get_dis(get_feature('{}-{}'.format(x1, y1)), get_feature('{}-{}'.format(x2, y2))))
#         ans[j][i] = ans[i][j]
#
# with open('result.txt', 'w') as f:
#     for i in range(50):
#         for j in range(50):
#             f.write('%.lf ' % (ans[i][j]))
#         f.write('\n')
# with open('result.csv', 'w', newline='') as f:
#     w = csv.writer(f)
#     for i in range(50):
#         w.writerow(ans[i])

with open('result.csv', 'r') as f:
    r = csv.reader(f)
    data = []
    for x in r:
        data.append(x)

# points = []
# for i in range(10):
#     l, r = i * 5, (i + 1) * 5
#     min_point = l
#     min_value = 0
#     for j in range(l, r):
#         min_value += float(data[l][j])
#
#     for k in range(l+1, r):
#         tmp = 0
#         for j in range(l, r):
#             tmp += float(data[k][j])
#         if tmp < min_value:
#             min_value = tmp
#             min_point = k
#     points.append(min_point)
# print(points)

points = [1, 7, 13, 16, 24, 27, 34, 35, 42, 46]
tot = 0
ans = []
for i in range(50):
    if i not in points:
        tmp = 100000
        point = -1
        for j in points:
            if float(data[i][j]) < tmp:
                tmp = float(data[i][j])
                point = j
        if i // 5 == point // 5:
            tot += 1
    else:
        point = i
    ans.append(point)

print(tot, tot / 40.)
print(ans)
