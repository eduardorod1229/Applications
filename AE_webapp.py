# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 11:43:14 2021

@author: EduardoR
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


st.write("""
         # Monthly Adverse Event report
""")

ae_data = pd.read_csv('NOAEs_ImportTemplate.csv')
ae_data['start_date'] = pd.to_datetime(ae_data['start_date'])
ae_data['start_date'] = ae_data['start_date'].dt.date

ae_data['end_date'] = pd.to_datetime(ae_data['end_date'])
ae_data['end_date'] = ae_data['end_date'].dt.date



#with st.beta_container():
#    hist_values = np.histogram(ae_data['start_date'].dt.month, bins=15, range=(0,13))[0]
#    st.write('Adverse Event by grade of severity')
#    st.bar_chart(hist_values)
    
st.sidebar.header('Select The data to display')

input_sdate = st.sidebar.date_input('Select Start date')
input_edate = st.sidebar.date_input ('Select End date')
input_study = st.sidebar.selectbox('Only SAEs?', ('Yes','No'))


filtered_df = ae_data[ae_data['start_date'] > input_sdate &(ae_data['start_date'] < input_edate)]



sae_df = ae_data[ae_data['serious']==1]
sae_df = sae_df[['id','ae_term','ongoing','relationship']]

if st.checkbox('SAEs'):
    st.subheader('Displaying only SAEs')
    st.write(sae_df)
    fig, ax = plt.subplots()
    ax.hist(sae_df['ae_term'])
    st.pyplot(fig)
#Create the sidebar the first parameter in the slider is the minimum value, the 
#second parameter is the maximum value and the last is the default value
'''
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length',1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width',0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features


#Assign the input parameters to a variable 'df'

df = user_input_features()

st.subheader('User Input Parameters')

#write the df variable into the app (creates a printout of the dataframe)
st.write(df)


#import the dataset to use the randomforest classifier
iris = datasets.load_iris()
X = iris.data
Y = iris.target


#Built and compile the random forest classifier using the iris data
clf = RandomForestClassifier()
clf.fit(X, Y)

#the predictions are out of the user input
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

#Create a printout of the class label and probability

st.subheader('Prediction')
st.write(iris.target_names[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
'''