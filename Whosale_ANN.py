#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2 


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



"""#feature selection
best_features = SelectKBest(score_func=chi2, k = 5)
fit = best_features.fit(x,y)
df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(x.columns)
features_scores = pd.concat([df_columns,df_scores],axis = 1)
features_scores.columns = ['features','score'] #giving name to columns 
print(features_scores.nlargest(2,'score'))"""


x = df[['Fresh','Grocery','Milk','Frozen','Detergents_Paper','Delicassen','Region']]
y = df.iloc[:,0:1].values


#Spliting dataset into Train and Test set 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 0)



#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


#importing keras library
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

reg = Sequential()

#Adding input layer and first hidden layer
reg.add(Dense(units = 7, kernel_initializer = "he_normal",
              activation = 'relu',input_dim = 7))


#Adding input layer and first hidden layer
reg.add(Dense(units = 7, kernel_initializer = "he_normal",
              activation = 'relu'))



#Adding input layer and first hidden layer
reg.add(Dense(units = 1, kernel_initializer = "glorot_uniform",
              activation = 'sigmoid'))     

#compiling ANN
reg.compile(optimizer = 'Adamax', loss = 'binary_crossentropy',metrics = ['accuracy'])




#fitting ANN to training dataset
model_history = reg.fit(x_train,y_train, validation_split = 0.33, batch_size = 10,
                        epochs = 50)

print(model_history.history.keys())



#predicting the test set
y_pred = reg.predict(x_test)
y_pred = (y_pred > 0.5)





from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)


from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)



#Classification Matrix Visualization
import seaborn as sns
axes = sns.heatmap(cm, square=True,annot=True,fmt='d',cbar = True, cmap = plt.cm.GnBu)
ax = plt.axes()
ax.set_title('ANN')





















