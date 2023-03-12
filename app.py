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

h = st.selectbox('Height of the Substrate in mm (h)',(0.8,1.6,3.2))
fr = st.text_input("Resonant Frequency in GHz (fr)")
fl = st.text_input("Lower-cutoff Frequency in GHz (fl)")
fh = st.text_input("Higher-cutoff Frequency in GHz (fh)")
bw = st.text_input("Bandwidth in GHz (bw)")
fbw = st.text_input("Fractional bandwidth (fbw)")
g = st.selectbox('Inter-gap Element spacing in mm (g)',(0.25,0.375,0.5))


# code for Prediction
msg1 = list()
msg1.append("Track-length (d) is")
msg2 = list()
msg2.append("Track-width (s) is")

# creating a button for Prediction    
if st.button("Predict"):
    d_pred = model_d.predict([[h,fr,fl,fh,bw,fbw,g]])
    #st.success('Track-length (d) is')
    msg1.append(d_pred)
    st.success(msg1) 
    
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
    msg2.append(s_pred)
    st.success(msg2)                    
      






