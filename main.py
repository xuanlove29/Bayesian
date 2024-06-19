import numpy as np
import time
import torch
import read
from sklearn.metrics import accuracy_score

#初始化变量
class_num=10
features_len=28*28

# 二值化
def binaryzation(img:torch.Tensor)->torch.Tensor:
    bin_img=torch.where(img>50,1,0)
    return bin_img
# 训练
def Train(trainset, train_labels):
    prior_probability = torch.zeros(class_num)  # 预设先验概率  p(y = j)
    conditional_probability = torch.zeros((class_num, feature_len, 2))  # 预设条件概率  p(xk | y)

    # 计算先验概率及条件概率
    for i in range(len(train_labels)):  # 训练集标签的数量

        img = binaryzation(trainset[i])  # 图片二值化   对每一个图片进行二值化

        label = train_labels[i]  # 对每一张图片赋予标签

        prior_probability[label] += 1  # 先验概率，总要计算出训练集中同一种标签的数量，比如说是数字1的图片有多少张

        for j in range(feature_len):  # 对每一张图片的每一个像素进行操作，统计同一种标签下的图像中各像素取值的数量
            conditional_probability[label][j][img[j]] += 1  # 属于第i类的图像中像素j为0和1的数量

    prior_probability = torch.log((prior_probability + 1) / (len(trainset)))  # 先验概率平滑
    # 将概率归到[1,10000]
    for i in range(class_num):
        for j in range(feature_len):
            # 经过二值化后图像只有0，1两种取值
            pix_0 = conditional_probability[i][j][0]  # 得到属于第i类的图像中第j个像素像素值为0的数量
            pix_1 = conditional_probability[i][j][1]  # 得到属于第i类的图像中第j个像素像素值为1的数量

            # 计算0，1像素点对应的条件概率  后验概率平滑
            probalility_0 = (float(pix_0) + 1) / (float(pix_0 + pix_1) + 2) * 1000000  # 拉普拉斯平滑
            probalility_1 = (float(pix_1) + 1) / (float(pix_0 + pix_1) + 2) * 1000000

            conditional_probability[i][j][0] = np.log(probalility_0)  # 得到 P(Xk|y)
            conditional_probability[i][j][1] = np.log(probalility_1)

    return prior_probability, conditional_probability


# 计算概率
def calculate_probability(img, label):
    probability = prior_probability[label]  # 先验概率
    sum = 0
    for i in range(len(img)):  # 取log变为相加
        sum += int(conditional_probability[label][i][img[i]])
    probability = probability + sum
    return probability

# 预测
def Predict(testset, prior_probability, conditional_probability):
    predict = []

    for img in testset:

        # 图像二值化
        img = binaryzation(img)

        max_label = 0
        max_probability = calculate_probability(img, 0)

        for j in range(1, 10):
            probability = calculate_probability(img, j)

            if max_probability < probability:
                max_label = j
                max_probability = probability

        predict.append(max_label)

    return torch.tensor(predict)

# 主函数
if __name__ == '__main__':
    print('Start read data')

    time_1 = time.time()
    imgs = read.images('minist')  # 取出像素值
    labels = read.labels('ministlabel')  # 取出标签值
    time_2 = time.time()
    print('read data cost ', time_2 - time_1, ' second', '\n')

    print('Start training')
    prior_probability, conditional_probability = Train(imgs, labels)
    time_3 = time.time()
    print('training cost ', time_3 - time_2, ' second', '\n')

    test_imgs = read.images('mnist_test')
    test_labels = read.labels('minist_testlabel')  # 取出测试集标签值

    print('Start predicting')
    test_predict = Predict(test_imgs, prior_probability, conditional_probability)
    time_4 = time.time()
    print('predicting cost ', time_4 - time_3, ' second', '\n')

    score = accuracy_score(test_labels, test_predict)
    print("The accruacy socre is ", score)
