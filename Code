import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

titanic_data = pd.read_csv('train123.csv')
print(titanic_data.head())

print(titanic_data.shape)

print(titanic_data.info)

############################################# Data Cleaning ###############################################

#To check null value
print(titanic_data.isnull().sum())

#handling the missing values cabin, age,embarked


#droping cabin column because of 687 missing value
titanic_data=titanic_data.drop(columns='Cabin',axis=1)
print(titanic_data.shape)

#replace null age with mean age
titanic_data['Age'].fillna(titanic_data['Age'].mean(),inplace=True)
print(titanic_data['Age'].isnull().sum())

#embarked by mode value
print(titanic_data["Embarked"].mode())

print(titanic_data["Embarked"].mode()[0])
#replace with s
titanic_data['Embarked'].fillna(titanic_data["Embarked"].mode()[0],inplace=True)
print(titanic_data['Embarked'].isnull().sum())

#overall check
print(titanic_data.isnull().sum())

############################################# Data Analysis ###############################################

print(titanic_data.describe())

#finding survivals
print(titanic_data['Survived'].value_counts())

############################################# Data Visualization ###############################################

sns.set()

#countplot for survived
ax=sns.countplot(x='Survived',data=titanic_data)
ax.set_title('People died and survived')

#counplot for Sex
az=sns.countplot(x='Sex',data=titanic_data)

#countplot for survival on basis of sex
ay=sns.countplot(data=titanic_data, x='Sex',hue='Survived')
ay.set_title('survival on basis of sex')

#survuial based on class
a1=sns.countplot(data=titanic_data, x='Pclass',hue='Survived')
a1.set_title('survival on basis of sex')

#machinelearning###########################

#encoding categorical columns
#male=0
#female=1

print(titanic_data['Sex'].value_counts())

print(titanic_data['Embarked'].value_counts())

#converting
titanic_data.replace({'Sex':{'male':0,'female':1},'Embarked':{'S':0,'C':1,'Q':2}},inplace=True)
print(titanic_data.head())



#seperate feature and label label=survived
X=titanic_data.drop(columns=['PassengerId','Name','Ticket','Survived'],axis=1)
Y= titanic_data['Survived']
print(X)
print(Y)

#split the data into trainning and test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)
print(X.shape,X_test.shape,X_train.shape)

#Model training

#Logistic Rgression

model = LogisticRegression()

#train the model

model.fit(X_train,Y_train)

#model Evaluation

#aacuracy of model on training data
X_train_predict= model.predict(X_train)

print(X_train_predict)
training_data_acc=accuracy_score(Y_train,X_train_predict)
print('Accuracy score  of training data: ',training_data_acc)

#aacuracy on test data
X_test_predict=model.predict(X_test)
print(X_test_predict)
test_data_acc= accuracy_score(Y_test,X_test_predict)
print('accuracy score of test data:',test_data_acc )

plt.show()
