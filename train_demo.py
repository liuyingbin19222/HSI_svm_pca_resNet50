from pylab import *
from numpy import *
from scipy.io import loadmat
import random
import warnings
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import svm
from sklearn.decomposition import NMF
import math
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.cbook
from sklearn import preprocessing
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from scipy.optimize import nnls
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import heapq
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)   #  取消警告;

random_state = np.random.RandomState(0)
number = 16
# 加载数据;
def load_mat():
    m = loadmat("./data/Indian_pines_corrected.mat")
    n = loadmat("./data/Indian_pines_gt.mat")  # 加载 ground_truth;
    reorder_n = np.reshape(n["indian_pines_gt"], 145 * 145)  # ground_truth转换为一维数组;
    reorder_l = np.reshape(m["indian_pines_corrected"], (145 * 145, 200))  # ground_pines转换为二维矩阵;
    return reorder_n,reorder_l

# 数据去零:
def removeZero():
    reorder_l_list = []
    reorder_n_list = []
    reorder_n , reorder_l = load_mat()
    for i in range(len(reorder_n)):
        if(reorder_n[i] != 0):
            reorder_l_list.append(reorder_l[i])
            reorder_n_list.append(reorder_n[i])
    reorder_l_list = np.array(reorder_l_list)
    return reorder_n_list, reorder_l_list
# 分为测试集和训练集:
def get_train_test():
    reorder_n_list,reorder_l_list = removeZero()
    # np.random.shuffle(reorder_l_list)  fuck !!!!
    x_train, x_test, y_train, y_test = train_test_split(reorder_l_list, reorder_n_list, test_size=0.8,random_state=random_state)
    return x_train,x_test,y_train,y_test
def get_indian_pines_train_test():
    reorder_n_list,reorder_l_list = removeZero()

    countNum = []
    for i in range(16):
        countNum.append(0)
    for i in range(len(reorder_n_list)):
        if reorder_n_list[i] == 1:
            countNum[0] += 1
        if reorder_n_list[i] == 2:
            countNum[1] += 1
        if reorder_n_list[i] == 3:
            countNum[2] += 1
        if reorder_n_list[i] == 4:
            countNum[3] += 1
        if reorder_n_list[i] == 5:
            countNum[4] += 1
        if reorder_n_list[i] == 6:
            countNum[5] += 1
        if reorder_n_list[i] == 7:
            countNum[6] += 1
        if reorder_n_list[i] == 8:
            countNum[7] += 1
        if reorder_n_list[i] == 9:
            countNum[8] += 1
        if reorder_n_list[i] == 10:
            countNum[9] += 1
        if reorder_n_list[i] == 11:
            countNum[10] += 1
        if reorder_n_list[i] == 12:
            countNum[11] += 1
        if reorder_n_list[i] == 13:
            countNum[12] += 1
        if reorder_n_list[i] == 14:
            countNum[13] += 1
        if reorder_n_list[i] == 15:
            countNum[14] += 1
        if reorder_n_list[i] == 16:
            countNum[15] += 1
    return countNum

# 获取数组中最小的三个元素的index;
def getSmall(array):
    # min1,min2,min3 = 16
    result = []
    top3SmallNum = heapq.nsmallest(3,array)
    for i in range(len(array)):
        if(top3SmallNum[0] == array[i] ):
            result.append(i+1)
        if(top3SmallNum[1] == array[i]):
            result.append(i+1)
        if(top3SmallNum[2] == array[i]):
            result.append(i+1)
    return result

def get_flag(arr,index):
    for i in range(len(arr)):
        if(arr[i] == index):
            return True
    return False

## 返回50/15的数据集;
def get_data():

    result = getSmall(get_indian_pines_train_test())
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    countNum = [0,0,0]
    countNum_half = 13*[0]
    countNum_ = []
    count_Index = []  # 存储index;
    for i in range(1,17): # 1-16 ， 减去result的三个元素，还剩下13个;
        if(not get_flag(result,i)):
            countNum_.append(i)

    reorder_n_list, reorder_l_list_ = removeZero()
    # shuffle
    finalData = np.vstack((reorder_n_list,reorder_l_list_.T))
    finalData = finalData.T
    np.random.shuffle(finalData)
    finalData = finalData.T

    reorder_n_list = finalData[0]
    reorder_n_list = reorder_n_list.T
    reorder_l_list = []
    for i in range(1,len(finalData)):
        reorder_l_list.append(finalData[i])
    reorder_l_list = np.array(reorder_l_list)
    reorder_l_list = reorder_l_list.T

    for i in range(len(reorder_n_list)):
        for j in range(len(result)):
            if(reorder_n_list[i] == result[j] and countNum[j] <= 14):
                x_train.append(reorder_l_list[i])
                y_train.append(reorder_n_list[i])
                countNum[j] += 1
                count_Index.append(i)
    for i in range(len(reorder_n_list)):
        if(reorder_n_list[i] == countNum_[0] and countNum_half[0] <= 49):
            x_train.append(reorder_l_list[i])
            y_train.append(reorder_n_list[i])
            countNum_half[0]+=1
            count_Index.append(i)
        if(reorder_n_list[i] == countNum_[1] and countNum_half[1] <= 49):
            x_train.append(reorder_l_list[i])
            y_train.append(reorder_n_list[i])
            countNum_half[1]+=1
            count_Index.append(i)
        if(reorder_n_list[i] == countNum_[2] and countNum_half[2] <= 49):
            x_train.append(reorder_l_list[i])
            y_train.append(reorder_n_list[i])
            countNum_half[2]+=1
            count_Index.append(i)
        if(reorder_n_list[i] == countNum_[3] and countNum_half[3] <= 49):
            x_train.append(reorder_l_list[i])
            y_train.append(reorder_n_list[i])
            countNum_half[3]+=1
            count_Index.append(i)
        if(reorder_n_list[i] == countNum_[4] and countNum_half[4] <= 49):
            x_train.append(reorder_l_list[i])
            y_train.append(reorder_n_list[i])
            countNum_half[4]+=1
            count_Index.append(i)
        if(reorder_n_list[i] == countNum_[5] and countNum_half[5] <= 49):
            x_train.append(reorder_l_list[i])
            y_train.append(reorder_n_list[i])
            countNum_half[5]+=1
            count_Index.append(i)
        if(reorder_n_list[i] == countNum_[6] and countNum_half[6] <= 49):
            x_train.append(reorder_l_list[i])
            y_train.append(reorder_n_list[i])
            countNum_half[6]+=1
            count_Index.append(i)
        if(reorder_n_list[i] == countNum_[7] and countNum_half[7] <= 49):
            x_train.append(reorder_l_list[i])
            y_train.append(reorder_n_list[i])
            countNum_half[7]+=1
            count_Index.append(i)
        if(reorder_n_list[i] == countNum_[8] and countNum_half[8] <= 49):
            x_train.append(reorder_l_list[i])
            y_train.append(reorder_n_list[i])
            countNum_half[8]+=1
            count_Index.append(i)
        if(reorder_n_list[i] == countNum_[9] and countNum_half[9] <= 49):
            x_train.append(reorder_l_list[i])
            y_train.append(reorder_n_list[i])
            countNum_half[9]+=1
            count_Index.append(i)
        if(reorder_n_list[i] == countNum_[10] and countNum_half[10] <= 49):
            x_train.append(reorder_l_list[i])
            y_train.append(reorder_n_list[i])
            countNum_half[10]+=1
            count_Index.append(i)
        if(reorder_n_list[i] == countNum_[11] and countNum_half[11] <= 49):
            x_train.append(reorder_l_list[i])
            y_train.append(reorder_n_list[i])
            countNum_half[11]+=1
            count_Index.append(i)
        if (reorder_n_list[i] == countNum_[12] and countNum_half[12] <= 49):
            x_train.append(reorder_l_list[i])
            y_train.append(reorder_n_list[i])
            countNum_half[12] += 1
            count_Index.append(i)

    # 获取测试集;
    for i in range(len(reorder_n_list)):
        if(not get_flag(count_Index,i)):
            x_test.append(reorder_l_list[i])
            y_test.append(reorder_n_list[i])
    # x_train_left, y_train_left,x_train_right,y_train_right = train_test_split(x_train,y_train,test_size=0.8 ,random_state=random_state)
    # x_train = np.append(x_train_left,y_train_left,axis=0)
    # y_train = np.append(x_train_right,y_train_right,axis=0)
    # x_test_left, y_test_left,x_test_right,y_test_right = train_test_split(x_test,y_test,test_size=0.3 , random_state=random_state)
    # x_test = np.append(x_test_left,y_test_left,axis=0)
    # y_test = np.append(x_test_right,y_test_right,axis=0)
    return x_train , x_test, y_train, y_test

# NMF分解,NNLS得到H:
def useNMF_NNLS(r):
    # x_train, x_test, y_train, y_test = get_train_test()
    x_train, x_test , y_train,y_test = get_data()
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_train = x_train.T
    x_test = x_test.T
    new_model = NMF(n_components= r,init='random')
    x_train_W = new_model.fit_transform(x_train)
    train_H = []
    test_H = []
    for i in range(len(x_train.T)):
        train_H.append(nnls(x_train_W,x_train.T[i])[0])
    for i in range(len(x_test.T)):
        test_H.append(nnls(x_train_W,x_test.T[i])[0])
    train_H = np.array(train_H)
    test_H = np.array(test_H)
    return train_H, test_H,y_train, y_test

# 数据归一化 ， svm分类器进行训练;
def getStander(r):
    svc = SVC()
    parameters = [
        {
            'C': [50,100,200,400,800,1600,3200,6400,12800,2**8*100,2**9*100,2**10*100],
            'gamma': [5e-4,5e-5,5e-6,5e-7,5e-8,5e-9,5e-10,5e-11,5e-12,5e-13,5e-14,5e-15],
            'kernel': ['rbf']
        },
        {
            'C': [50,100,200,400,800,1600,3200,6400,12800,2**8*100,2**9*100,2**10*100],
            'kernel': ['linear']
        },
        {
            'coef0':[0.0, 1, 2, 3, 4, 5, 6, 7,8,9,10,11,12,13,14,15],
            'kernel':['sigmoid']
        },
        {
            'degree':[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
            'gamma':[ 5e-9,5e-10,5e-11,5e-12,5e-13,5e-14,5e-15,1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10],
            'kernel':['poly']
        }
    ]
    train_H , test_H, y_train, y_test = useNMF_NNLS(r)
    scaler = StandardScaler()
    scaler.fit(train_H)
    train_H = scaler.transform(train_H)
    test_H = scaler.transform(test_H)
    # clf = AdaBoostClassifier(n_estimators=500, random_state=42)
    # clf = SVC(C=1e9,gamma=1e-7,kernel=kernel)
    # clf = KNeighborsClassifier(n_neighbors=i)
    # clf = DecisionTreeClassifier(random_state=42)
    # scores = cross_val_score(clf,train_H,y_train,cv=i)
    # print("模型平均scores:",scores.mean())
    clf = GridSearchCV(svc,parameters,cv=5,n_jobs=-1)
    clf.fit(train_H , y_train)
    y_pred = clf.predict(test_H)
    print(classification_report(y_test,y_pred))
    return precision_score(y_test,y_pred,average="micro")

def getStander_Knn(r):
    num = []
    for i in range(2,16):
        train_H , test_H, y_train, y_test = useNMF_NNLS(r)
        scaler = StandardScaler()
        scaler.fit(train_H)
        train_H = scaler.transform(train_H)
        test_H = scaler.transform(test_H)
        # clf = AdaBoostClassifier(n_estimators=500, random_state=42)
        # clf = SVC(C=1e8,gamma=1e-7,kernel=kernel)
        clf = KNeighborsClassifier(n_neighbors=i)
        # clf = DecisionTreeClassifier(random_state=42)
        scores = cross_val_score(clf,train_H,y_train,cv=i)
        num.append(scores.mean())
        print("模型平均scores:",scores.mean())
    print("scores的最大值索引:",np.argmax(num))
    clf = KNeighborsClassifier(n_neighbors=np.argmax(num))
    clf.fit(train_H , y_train)
    y_pred = clf.predict(test_H)
    print(classification_report(y_test,y_pred))
    return precision_score(y_test,y_pred,average="micro")
def getStander_(kernel):
    #未使用NMF分解;
    train_H, test_H , y_train,y_test = get_data()
    scaler = StandardScaler()
    scaler.fit(train_H)
    train_H = scaler.transform(train_H)
    test_H = scaler.transform(test_H)
    # clf = AdaBoostClassifier(n_estimators=500, random_state=42)
    clf = SVC(C=1e8,gamma=1e-7,kernel=kernel)
    # clf = DecisionTreeClassifier(random_state=42)
    clf.fit(train_H,y_train)        # 训练分类器;
    y_pred = clf.predict(test_H)
    print(classification_report(y_test,y_pred))
    return precision_score(y_test,y_pred,average="micro")
if __name__ == "__main__":
    scores = []
    kernel_func = ["rbf","sigmoid","linear"]
    # for k in kernel_func:
    #     getStander_(k)
    # for i in range(10,210,10):
    #     print("次数:",i)
    #     getStander_Knn(i)
    # for k in kernel_func:
    for i in range(10,220,10):
        # getStander()
        scores.append(getStander(i))
        # print("precision_score:",getStander(i))
        print("次数:", i)
    # 画图部分，可删去
    x = range(10,220,10)
    y = scores
    plt.title("precision-score:")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x,y)
    plt.savefig("./t2.png")
    plt.show()

