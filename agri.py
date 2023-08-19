#importing libraries
import numpy as np
import pandas as pd

# Import Libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns
     

#importing data set using pandas
from google.colab import files 
uploaded = files.upload()
import io
agri_data = pd.read_csv('C:/Users/Kummera Sneha/OneDrive/Desktop/csv.csv')


print(agri_data.shape)
agri_data.head()
agri_data.info()

c_data = agri_data.copy()
c_data.columns

#checking null value
c_data.isnull().sum()
price_data = c_data[['Price Date','Min Price (Rs./Quintal)','Max Price (Rs./Quintal)','Modal Price (Rs./Quintal)']]
#printing price data    TASK (A)
price_data
c_data
plt.figure(figsize=(10,5))
c_data.groupby(['Market Name'])['Modal Price (Rs./Quintal)'].count().sort_values(ascending=False).head(10).plot.bar()
plt.ylabel('Price of potatoes')
plt.title("Market comparision",fontsize=20)
plt.show()
#analyzing minimuim and maximuim value of potatoes in year
df1 = agri_data.copy()
df1.sort_values('Min Price (Rs./Quintal)',inplace=True)
df1.head(20)
df1.tail()
# Import libraries for train test split
from sklearn.model_selection import train_test_split

# import Ilbrary for Scaling
from sklearn.preprocessing import StandardScaler

# import Ilbrary for Model Building
from sklearn.linear_model import LinearRegression

     

#creating copy of data
ml_data = agri_data.copy()
     

#Applying mean encoding for Market name
name = ml_data.groupby('Market Name')['Modal Price (Rs./Quintal)'].mean()
ml_data['Market Name'] = ml_data['Market Name'].map(name)

print(ml_data['Market Name'])
#dropping unwanted columns
ml_data = ml_data.drop(['Sl no.','District Name','Commodity','Variety','Grade','Price Date'],axis = 1)
ml_data.info()
agri_train,agri_test = train_test_split(ml_data,train_size = 0.7,random_state = 42)
     

print(agri_train.shape)
print(agri_test.shape)
# Divide tarin set into Dependent and independent variables
y_train = agri_train.pop('Modal Price (Rs./Quintal)')

X_train = agri_train


# Divide test set into Dependent and independent variables
y_test = agri_test.pop('Modal Price (Rs./Quintal)')

X_test = agri_test
     

# Scale the train
scaler = StandardScaler()

X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])

X_train.describe()
# Scale the test
X_test[X_test.columns] = scaler.transform(X_test[X_test.columns])

X_test.describe()
# Build the model
lr = LinearRegression()

agri_reg = lr.fit(X_train,y_train)
     

# the r2 score
agri_reg.score(X_train,y_train)
# r2 for test data
agri_reg.score(X_test,y_test)
# Plot Distribution plot of Residuals
plt.figure(figsize=(10,5))
y_train_pred = agri_reg.predict(X_train)
res = y_train - y_train_pred
sns.distplot(res)
plt.xlabel('Residuals')
plt.title("Residual Analysis",fontsize=20)
plt.show()
sns.scatterplot(x=res,y=y_train_pred)
plt.xlabel('Residuals')
plt.title("Residual Analysis",fontsize=20)
plt.show()
# Print coef
print("Coef are:",agri_reg.coef_)

#print intercept
print("Intercept is",agri_reg.intercept_)
model = str(agri_reg.intercept_)

for i in range(len(agri_reg.coef_)):
    model = model +' + '  +(str(agri_reg.coef_[i])) + ' * ' +(str(X_train.columns[i]))
print(model)