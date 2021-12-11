import random
from libsvm.svm import RBF
from libsvm.svmutil import svm_predict
from libsvm.svmutil import svm_problem
from libsvm.svmutil import svm_train
from libsvm.svmutil import svm_parameter

CLASSES = ['Iris-setosa','Iris-versicolor','Iris-virginica']
RBF = 2

def main():

    train_set,test_set = splitData()
    print(test_set)
    results = None
    models = getModels(train_set)
    print(models)
    print("done")
    results = classify(models,test_set)
    totalCount = 0
    totalCorrect = 0
    for clazz in CLASSES:
        count,correct = results[clazz]
        totalCount += count
        totalCorrect += correct
        print("%s %d %d %f"%(clazz,correct,count,(float(correct)/count)))
    print("%s %d %d %f"%("Overall",totalCorrect,totalCount,(float(totalCount)/totalCount)))

def classify(models,dataSet):
    results = {}
    for trueClass in CLASSES:
        count = 0
        correct = 0
        for item in dataSet[trueClass]:
            predClass,prod = predict(models,item)
            print("trueClass=%s, predClass=%s, prob=%f"%(trueClass,predClass,prod))
            count += 1
            if trueClass == predClass: correct+=1
        results[trueClass] =(count,correct)
    return results

def predict(models,item):
    maxProb = 0.0
    bestClass = ""
    for clazz,model in models.items():
        prob = predictSingle(model,item) 
        if prob>maxProb:
            maxProb = prob
            bestClass = clazz
    return (bestClass,maxProb)

def predictSingle(model,item):
    output = svm_predict([0],[item],model,"-q -b 1")
    prob = output[2][0][0]
    return prob
    


def getModels(train):
    models = {}
    param = getParam()
    for c in CLASSES:
        labels,data = getTariningData(train,c)
        prob = svm_problem(labels,data)
        m = svm_train(prob,param)
        models[c] = m
    return models
        
    
def getTariningData(trainData,clazz):
    ld_list  = []
    labeledData = getLabeledDataVector(trainData,clazz,1)
    ld_list.append(list(labeledData))
    negClasses = [c for c in CLASSES if not c == clazz]
    for c in negClasses:
        ld = getLabeledDataVector(trainData,c,-1)
        ld_list.append(list(ld))

    random.shuffle(ld_list)
    labels = []
    data = []
    for ele in ld_list:
        for two_tuple in ele:
            labels.append(two_tuple[0])
            data.append(two_tuple[1])

    labels = tuple(labels) 
    data = tuple(data)
    return (labels,data)

def getLabeledDataVector(trainSet,clazz,label):
    data = trainSet[clazz]
    labels = [label]*len(data)
    output = zip(labels,data)
    return output
    
def getParam():
    parm = svm_parameter("-q")
    parm.probability = 1
    parm.kernel_type = RBF
    parm.C = .77777777
    parm.gamma = .000000001

    return parm

def splitData():
    f = open("./iris.data",mode='r',encoding='utf8')
    lines = f.readlines()
    data_list = []
    for line in lines:
        if len(line)>2:
            data_list.append(lineStrToFloat(line.strip().split(','))) 
    trainData = {}
    testData = {}
    setosa = []
    versicolor = []
    virginica = []
    for ele in data_list:
        if ele[4]=='Iris-setosa':
            setosa.append(ele[:4])
        elif ele[4]=='Iris-versicolor':
            versicolor.append(ele[:4])
        elif ele[4]=='Iris-virginica':
            virginica.append(ele[:4])
    print(len(setosa))
    
    count = int(.7*len(setosa))
    trainData[CLASSES[0]]=setosa[:count]
    testData[CLASSES[0]]=setosa[count:]
    trainData[CLASSES[1]]=versicolor[:count]
    testData[CLASSES[1]]=versicolor[count:]
    trainData[CLASSES[2]]=virginica[:count]
    testData[CLASSES[2]]=virginica[count:]
    return trainData , testData 


def lineStrToFloat(line_str):
    line_con = []
    for x in range(len(line_str)-1):
        line_con.append(float(line_str[x]))
        if x == len(line_str)-2:
            line_con.append(line_str[x+1])
    return line_con
    
if __name__ == '__main__':
    main()
