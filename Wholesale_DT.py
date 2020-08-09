import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2 


df = pd.read_csv('Wholesale.csv',index_col = False)
x = df.iloc[:,1:8]
y = df.iloc[:,-1].values


#feature selection
best_features = SelectKBest(score_func=chi2, k = 5)
fit = best_features.fit(x,y)
df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(x.columns)

features_scores = pd.concat([df_columns,df_scores],axis = 1)
features_scores.columns = ['features','score'] #giving name to columns 
print(features_scores.nlargest(5,'score'))


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1, random_state = 0)



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.tree import DecisionTreeClassifier
reg = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)


from sklearn.metrics import confusion_matrix
con = confusion_matrix(y_test,y_pred)


from matplotlib.colors import ListedColormap
x_set,y_set = x_train,y_train
x1, x2 = np.meshgrid(np.arange(start = x_set[:,0].min() -1, stop = x_set[:,0].max() +1,step = 0.01),
                     np.arange(start = x_set[:,1].min() -1,stop = x_set[:,1].max() +1,step = 0.01))
plt.contourf(x1,x2, reg.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red','green')))
plt.xlim(x1.min(), x1.max())
plt.xlim(x2.min(), x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j,0],x_set[y_set ==j, 1],
                c = ListedColormap(('red','green'))(i),label = j)
    
plt.title('Support Vector Machine')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()






















"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('Wholesale.csv',index_col = False)


x = df.iloc[:,[4,7]].values
y = df.iloc[:,0].values


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1, random_state = 0)



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 15)
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)




from sklearn.metrics import confusion_matrix
con = confusion_matrix(y_test,y_pred)


from matplotlib.colors import ListedColormap
x_set,y_set = x_train,y_train
x1, x2 = np.meshgrid(np.arange(start = x_set[:,0].min() -1, stop = x_set[:,0].max() +1,step = 0.01),
                     np.arange(start = x_set[:,1].min() -1,stop = x_set[:,1].max() +1,step = 0.01))
plt.contourf(x1,x2, knn.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red','green')))
plt.xlim(x1.min(), x1.max())
plt.xlim(x2.min(), x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j,0],x_set[y_set ==j, 1],
                c = ListedColormap(('red','green'))(i),label = j)
plt.title('Random Forest Classifier')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()"""