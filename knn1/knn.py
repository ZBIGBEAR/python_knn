#该程序实现了kNN算法
import numpy
import random
import operator
import matplotlib.pyplot as plt
plt.switch_backend('agg')
axes = plt.subplot(111)

def kNNClassifier(inX, dataSet, labels, k):
    '''
    inX: 被分类的测试样本（向量形式）；
    dataSet: 训练样本集，每个样本为一行，则行数=样本个数
    labels：每个样本的标记，list形式
    k：近邻个数
   
    sortedClassCount：返回字典，key是排名，value是标记
    '''
    dataSetSize=dataSet.shape[0]  #shape函数返回行数，也就是训练样本个数
    diffMat=numpy.tile(inX, (dataSetSize, 1))-dataSet   #tile复制测试向量，dataSetSize行，1列，为了相减
    sqDiffMat=diffMat**2  #差值求平方
    sqDistances=sqDiffMat.sum(axis=1)  #按每行求和
    distances=sqDistances**0.5  #开方，得到测试样本和每个训练样本（每行）的距离
    sortedDistIndices=distances.argsort()  #距离排序
   
    classCount={}  #key：样本标记，value：得票数
    for i in range(k):  #计数k个近邻样本的标记
        voteIlabel=labels[sortedDistIndices[i]]  #voteIlabel记录第i个样本的标记
        classCount[voteIlabel]=classCount.get(voteIlabel, 0)+1  #计票
    sortedClassCount=sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  #按照每个样本标记的得票数，降序排序
   
    return sortedClassCount[0][0]  #返回排名第一的标记，就当作测试样本的标记



if __name__=="__main__":
    dataSet=numpy.array([[1., 1.1],  #每行是一个样本，这点和MATLAB中实现代码不同
                   [1., 1.],
                   [0., 0.],
                   [0., 0.1]])
    labels=['A', 'A', 'B', 'B']  #样本的标记，行向量，一个元素代表一个样本的标记
    k=3  #近邻范围
    Axx = [1,1]
    Ayy = [1.1,1]   
    Bxx = [0,0]
    Byy = [0,0.1]
    #inX=[0, 0]  #测试样本
    N = 1000
    for i in range(N):
        x = random.random()
        y = random.random()
        inX = [x,y]
        #xx.append(x)
        #yy.append(y)
        classifyingResult=kNNClassifier(inX, dataSet, labels, k)  #调用KNN分类器，得到分类结果
        ret = str(classifyingResult)
        if ret=='B':
            Bxx.append(x)
            Byy.append(y)
        else:
            Axx.append(x)
            Ayy.append(y)
        #print( '['+str(x)+','+str(y)+']'+'classifyingResult = '+str(classifyingResult))  #输出分类结果
    type1 = axes.scatter(Axx,Ayy,s=20,c='red')
    type2 = axes.scatter(Bxx,Byy,s=20,c='green')
    axes.legend((type1,type2),(u'A',u'B'))
    plt.savefig(str(N)+'.png')
