import cv2
import torch
from torchvision import transforms

from lab3.train.train_net import Net


dictionary = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
              'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
              'W', 'X', 'Y', 'Z', '京', '闽', '粤', '苏', '沪', '浙']


def detection(img):
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    model = Net()
    model.load_state_dict(torch.load('../model/model.pkl'))

    left = True
    cols = []
    x = 0
    for i in range(img.shape[1]):
        if left and sum(img[:, i]) > 0:
            left = False
            x = i
        elif not left and sum(img[:, i]) == 0:
            cols.append([x, i])
            left = True

    rows = []
    for i, item in enumerate(cols):
        tmp = img[:, item[0]: item[1]+1]

        x = 0
        top = True
        for j in range(img.shape[0]):
            if top and sum(tmp[j, :]) > 0:
                top = False
                x = j
            elif not top and sum(tmp[j, :]) == 0 and j > img.shape[0] / 2:
                rows.append([x, j])
                break

    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ])

    string = ''
    for i in range(len(cols)):
        img_tmp = img[rows[i][0]: rows[i][1] + 1, cols[i][0]: cols[i][1] + 1]
        img_tmp = cv2.resize(img_tmp, (32, 40))
        tmp = trans(img_tmp)
        tmp = tmp.unsqueeze(0)
        out = model(tmp)
        pred = out.max(1, keepdim=True)[1]
        string += dictionary[pred]
    return string
