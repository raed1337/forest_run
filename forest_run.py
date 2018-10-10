import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


url = "C:/Users/kirito/Desktop/train.csv"
# load dataset into Pandas DataFrame
train = pd.read_csv(url)
Cover_type_data = pd.DataFrame(
    {'Cover_Type':[1,2,3,4,5,6,7],
     'description':['Spruce/Fir','Lodgepole Pine','Ponderosa Pine','Cottonwood/Willow','Aspen','Douglas-fir','Krummholz']
     })

train = pd.merge(train , Cover_type_data ,on='Cover_Type',how='inner')
train =pd.DataFrame(train)

train_x = train.drop(['Cover_Type','description'],axis=1)
names=train_x.columns.values.tolist()
train_y = train[['Cover_Type','description']]

train_x = StandardScaler().fit_transform(train_x)
train_x = pd.DataFrame(train_x , columns=names)


pca = PCA(n_components=2)
principalComponents = pca.fit_transform(train_x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, train_y['description']], axis = 1)

plt.scatter(principalDf['principal component 1'], principalDf['principal component 2'],
            c=train_y['Cover_Type'], edgecolor='none', alpha=0.5)
plt.set_cmap('Accent')
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar();
plt.show()

url_test = "C:/Users/kirito/Desktop/test.csv"

test_x = pd.read_csv(url_test)
print(test_x.head(100))

X_train, X_Cross_Validation, y_train, y_Cross_Validation = train_test_split(train_x,train_y,test_size=0.2,random_state=4)

lr=LogisticRegression(solver='lbfgs')
lr.fit(X_train , y_train['Cover_Type'])
lr.predict(X_Cross_Validation)
score = lr.score(X_Cross_Validation,y_Cross_Validation['Cover_Type'])
print(pd.DataFrame(lr.predict_proba(X_Cross_Validation)))

