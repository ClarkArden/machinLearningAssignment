import torch
import numpy as np
import random

a_np = np.array(range(1000))
#x = torch.rand(100, 28, 28)
#y = torch.randn(100, 28, 28)
#x = torch.cat((x, y), dim=0)
x = torch.from_numpy(a_np)
label = [1]*500+[0]*500
label = torch.tensor(label, dtype=torch.long)

index = [i for i in range(len(x))]
random.shuffle(index)
x = x[index]
label = label[index]

def get_k_fold_data(k, i, x, y):
    """TODO: Docstring for get_k_fold_data.

    :k:分成k份
    :i:取i份为数据为验证数据
    :x:原始数据
    :y:标签数据
    :returns: 返回训练数据xtrain，y_train,以及验证数据x_valid,y_valid
    @auther:刘皓皎
    """

    assert k > 1
    fold_size = x.shape[0]//k
    x_train, y_train = None, None
    x_valid, y_valid = None, None
    for j in range(k):
        idx = slice(j*fold_size, (j+1)*fold_size)
        x_part, y_part = x[idx], y[idx]
        if j == i:
            x_valid, y_valid = x_part, y_part
        elif x_train is None:
            x_train, y_train = x_part, y_part
        else:
            x_train = torch.cat((x_train, x_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)

    print(x_train.shape, y_train.shape)
    print(x_valid.shape, y_valid.shape)
    return x_train, y_train, x_valid, y_valid


def k_fold(k, x_train, y_train):
    for i in range(k):
        train_featurn, train_label, test_featurn, test_label = get_k_fold_data(
            k, i, x_train, y_train)
        
        print("id of train data",train_featurn)
        print("第i折的训练集标签为",train_label)
        print("id of valid data",test_featurn)
        print("测试集的标签为",test_label)
k_fold(10,x,label)

print(type(index))
print(index)


