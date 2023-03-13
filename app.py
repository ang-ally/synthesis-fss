# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 21:18:02 2023

@author: Angelyn Sweety Isaac
"""

import pickle
import pandas as pd
from sklearn.preprocessing import RobustScaler
import streamlit as st

model_d = pickle.load(open('d_prediction.sav','rb'))
model_s = pickle.load(open('s_prediction.sav','rb'))

st.title("Synthesis of Square Loop FSS using ML")
st.markdown(
   f"""
   <style>
   p {
   background-image: url(‘bg.jpg’);
   }
   </style>
   """,
   unsafe_allow_html=True)
st.header('For FR-4 Substrate')

st.caption('The Range for Resonant Frequency: 0 < fr < 5') 
st.caption('The Range for Lower-cutoff Frequency: 0 < fl < 5')
st.caption('The Range for Higher-cutoff Frequency: 0 < fh < 5')

h = st.selectbox('Height of the Substrate in mm (h)',(0.8,1.6,3.2))
fr = st.number_input("Resonant Frequency in GHz (fr)",min_value=0.01,max_value=5.00,step=0.01)
fl = st.number_input("Lower-cutoff Frequency in GHz (fl)",min_value=0.01,max_value=5.00,step=0.01)
fh = st.number_input("Higher-cutoff Frequency in GHz (fh)",min_value=0.01,max_value=5.00,step=0.01)
bw = fh - fl
fbw = bw / fr
g = st.selectbox('Inter-gap Element spacing in mm (g)',(0.25,0.375,0.5))


# code for Prediction
msg1 = list()
msg1.append("Track-length (d) is")
msg2 = list()
msg2.append("Track-width (s) is")

# creating a button for Prediction    
if st.button("Predict"):
    col1, col2 = st.columns(2)
    
    d_pred = model_d.predict([[h,fr,fl,fh,bw,fbw,g]])
    col1.metric(label="Track Length in mm is : ",value=d_pred)
    
    df=pd.read_excel('final fr4 ds.xlsx')
    X=df[['h','fr','fl','fh','bw','fbw','g']]
    Y=df[['s']]
    scale_in=RobustScaler()
    scale_out=RobustScaler()
    x=scale_in.fit_transform(X)
    y=scale_out.fit_transform(Y)
    prediction=scale_in.transform([[h,fr,fl,fh,bw,fbw,g]])
    s_pred=model_s.predict(prediction)
    s_pred=s_pred.reshape(-1,1)
    s_pred=scale_out.inverse_transform(s_pred)
    s_pred=s_pred.reshape(1,-1) 
    col2.metric(label="Track Width in mm is: ",value=s_pred)  

                 
      






