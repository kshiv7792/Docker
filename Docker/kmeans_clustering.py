import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from sklearn.cluster import	KMeans
import pickle
# from scipy.spatial.distance import cdist 

# Generating random uniform numbers 
X = np.random.uniform(0,1,50)
Y = np.random.uniform(0,1,50)
df_xy = pd.DataFrame(columns=["X","Y"])
df_xy.X = X
df_xy.Y = Y

df_xy.plot(x="X", y="Y", kind = "scatter")

model1 = KMeans(n_clusters = 3).fit(df_xy)
#mywork
b=model1.cluster_centers_
b[:,1]
df_xy.plot(x = "X", y = "Y", c = model1.labels_, kind="scatter", s = 10, cmap = plt.cm.coolwarm)
#mywork
plt.plot(b[:,0],b[:,1],'o','black')
# Kmeans on University Data set 
Univ1 = pd.read_excel("C:/Users/kiran/Desktop/360 DIGI/University_Clustering.xlsx")

Univ1.describe()
Univ = Univ1.drop(["State"], axis = 1)

# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

from sklearn.preprocessing import MinMaxScaler
min_max = MinMaxScaler()
min_max.fit(Univ.iloc[:, 1:])    
maxe= min_max.transform(Univ.iloc[:, 1:])

pickle.dump(min_max, open('min_max.pkl', 'wb'))

#model=pickle.load(open('mpg.pkl', 'rb'))
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open('min_max.pkl', 'rb'))

model.fit(maxe)
#a = model.predict(min_max.transform([[1,2,3,4,90,6]]))
#model.labels_
# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(Univ.iloc[:, 1:])

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)


TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
Univ['clust'] = mb # creating a  new column and assigning it to new column 
pickle.dump(model, open('model.pkl', 'wb'))
Univ.head()
df_norm.head()

Univ = Univ.iloc[:,[7,0,1,2,3,4,5,6]]
Univ.head()

Univ.iloc[:, 2:8].groupby(Univ.clust).mean()

#Univ.to_csv("Kmeans_university.csv", encoding = "utf-8")

import os
os.getcwd()
