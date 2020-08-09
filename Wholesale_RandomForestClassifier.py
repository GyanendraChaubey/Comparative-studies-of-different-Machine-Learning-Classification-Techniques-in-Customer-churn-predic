#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



#Importing Dataset
df = pd.read_csv('Wholesale.csv',index_col = False)
x = df.iloc[:,1:8]
y = df.iloc[:,0:1].values



#Data Visualization
us = df[df.Channel == 1]
chi = df[df.Channel == 2]
plt.scatter(us.Grocery,us.Detergents_Paper,s = 5)
plt.scatter(chi.Grocery,chi.Detergents_Paper, s = 5)
plt.xlabel('Grocery')
plt.ylabel('Detergents_paper')
plt.legend(['Horeca','Retail'])
plt.show()



#feature selection
best_features = SelectKBest(score_func=chi2, k = 5)
fit = best_features.fit(x,y)
df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(x.columns)
features_scores = pd.concat([df_columns,df_scores],axis = 1)
features_scores.columns = ['features','score'] #giving name to columns 
print(features_scores.nlargest(2,'score'))



#Spliting dataset into Train and Test set 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 0)



#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)



#Handling the imbalance dataset
from imblearn.combine import SMOTETomek
smk= SMOTETomek()
x_res,y_res = smk.fit_sample(x,y)



#Building Model
reg = RandomForestClassifier()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)



#confusion matrix
from sklearn.metrics import confusion_matrix
con = confusion_matrix(y_test,y_pred)
print(con)



#checking Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))



#Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))




"""#ROC and AUC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
ras = roc_auc_score(y_test, reg.predict(x_test))
fpr,tpr,threshold = roc_curve(y_test,reg.predict_proba(x_test))
plt.figure()
plt.plot(fpr,tpr, label = 'xyz')
plt.show()"""



#Classification Matrix Visualization
import seaborn as sns
axes = sns.heatmap(con, square=True,annot=True,fmt='d',cbar = True, cmap = plt.cm.GnBu)
ax = plt.axes()
ax.set_title('Random_Forest_Classifier')





"""
#Data visualization
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
plt.legend()
plt.show()"""
