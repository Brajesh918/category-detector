import pandas as pd 
import numpy as np
from time import time
import nltk
import seaborn as sb
from matplotlib import pyplot as plt
import re
#reading data 
data=pd.read_csv(r"topic_classification_data.csv")
#print("total sample is  {}".format(data.shape))
#cat_information=data["Category"].value_counts()
#len(cat_information)
sb.countplot("Category",data=data)
plt.xticks(rotation=90)
plt.show()
#cleaning data
def tex_cleaning(data):
    corpus=[]
    for i in range(0,len(data)):
        process_data=re.sub(r'\W',' ',str(data[i]))
        process_data=process_data.lower()
        process_data=re.sub(r'\s+',' ',process_data)
        process_data=re.sub(r'\d+',' ',process_data)
        process_data=re.sub(r'[^a-zA-Z]',' ',process_data)
        corpus.append(process_data)
    return corpus
corpus=tex_cleaning(data["Desc"])
#corpus
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords 
tf_Vector=TfidfVectorizer(max_features=len(corpus),min_df=1,max_df=.8,stop_words=stopwords.words('english'))
#type(tf_Vector)
tf_Vector_matrix=tf_Vector.fit_transform(corpus).todense()
tf_Vector_matrix
tf_name=tf_Vector.get_feature_names()
x_feature=tf_Vector_matrix
y_cat=data.Category
#train model
x_train,x_test,y_train,y_test=train_test_split(x_feature,y_cat,test_size=30,random_state=10)
time_s=time()
knn_model=KNeighborsClassifier()
knn_learner=knn_model.fit(x_train,y_train)
time_end=time()
time_knn=time_end-time_s
print("Time taken by knn model is {}".format(time_knn))
#testing
time_s=time()
Yp=knn_learner.predict(x_test)
acc=accuracy_score(Yp,y_test)
time_end=time()
print("acc is {} and time taken is {}".format(acc,(time_end-time_s)))
#testing with data
test_keyword="teacher"
testInput=tf_Vector.transform([test_keyword])
knn_learner.predict(testInput)
#MODEL NB
time_s=time()
nb_model=MultinomialNB()
nb_learner=nb_model.fit(x_train,y_train)
time_end=time()
nb_train_time=time_end-time_s
print("time taken by nb model{}".format(nb_train_time))
#Example
testInput=tf_Vector.transform(["apple"])
nb_learner.predict(testInput)
#output Fruit
#saving the model
filename = 'ccp.pkl'
joblib.dump(model-learner, filename)