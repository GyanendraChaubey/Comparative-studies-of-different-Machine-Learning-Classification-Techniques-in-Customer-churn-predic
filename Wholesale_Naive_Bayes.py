#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score



#Importing Dataset
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
print(features_scores.nlargest(7,'score'))

x = df[['Fresh','Grocery','Milk','Frozen','Detergents_Paper','Delicassen','Region']]
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



#Spliting dataset into Train and Test set 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 0)



#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)



"""#Handling the imbalance dataset
from imblearn.combine import SMOTETomek
smk= SMOTETomek()
x_res,y_res = smk.fit_sample(x,y)"""



#Building Model
#K_Nearest
reg = GaussianNB()
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




#ROC and AUC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
clf_probs = reg.predict_proba(x_test)
clf_probs = clf_probs[:,1]
print(clf_probs)
ras = roc_auc_score(y_test,clf_probs)
print("Logistic : ROC AUC = %.3f" %(ras))
from sklearn.preprocessing import label_binarize
y = label_binarize(y_test,classes = [1,2])
n_classes = y.shape[1]
fpr,tpr,_ = roc_curve(y,clf_probs)
plt.figure()
lw = 2
plt.plot(fpr,tpr, color = "orange", lw = lw, label = "ROC curve (area = %0.2f" % ras)
plt.plot([0,1],[0,1], color = "blue",lw = lw, linestyle = '--')
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.05)
plt.xlabel('False Positive Value')
plt.ylabel('True Positive Value')
plt.title('Receiver operating Characteristics')
plt.legend(loc = "lower right")
plt.show()






#Classification Matrix Visualization
import seaborn as sns
axes = sns.heatmap(con, square=True,annot=True,fmt='d',cbar = True, cmap = plt.cm.GnBu)
ax = plt.axes()
ax.set_title('K-nearest')

