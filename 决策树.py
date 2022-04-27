import numpy as np
from sklearn.model_selection import GridSearchCV
import xlrd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.tree as sklTree
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def excel2matrix(path):
    data = xlrd.open_workbook(path)
    table = data.sheets()[0]
    nrows = table.nrows  # 行数
    ncols = table.ncols  # 列数
    datamatrix = np.zeros((nrows, ncols))
    for i in range(ncols):
        cols = table.col_values(i) #获取整列数值
        datamatrix[:, i] = cols
    return datamatrix

pathX = 'imf2.xlsx'  #  113.xlsx 在当前文件夹下
x1 = excel2matrix(pathX)

data=x1[:,:5]
data_2=x1[:,5:]

df=pd.DataFrame(data[:],columns=['a','b','c','d','e'])
x = df['c']
y = df['e']
z = df['d']
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x[:100],y[:100],z[:100],c='red')
ax.scatter(x[100:],y[100:],z[100:],c='blue')

ax.set_xlabel('x',fontdict={'size':10,'color':'red'})
ax.set_ylabel('y',fontdict={'size':10,'color':'red'})
ax.set_zlabel('z',fontdict={'size':10,'color':'red'})
plt.show()

X_train, X_test, y_train, y_test = train_test_split(data, data_2, test_size=0.2,random_state=0)


tree_param_grid = {'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10],'max_depth':[2,3,4,]}
grid = GridSearchCV(sklTree.DecisionTreeClassifier(),param_grid=tree_param_grid,cv=5)
grid.fit(X_train,y_train)
print('best_params_',grid.best_params_)

#训练
decisionTree = sklTree.DecisionTreeClassifier(min_samples_leaf=10,max_depth = 3);
decisionTree.fit(X_train,y_train);  #训练



fpr, tpr, threshold = metrics.roc_curve(y_test, decisionTree.predict_proba(X_test)[:,1])  ###计算真正率和假正率
print(fpr)
print(tpr)
print(threshold)
roc_auc = auc(fpr, tpr)  ###计算auc的值，auc就是曲线包围的面积，越大越好

plt.figure()
lw = 2
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()