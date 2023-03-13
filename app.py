import pickle
import pybase64
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import streamlit as st

model_d = pickle.load(open('d_prediction.sav','rb'))
model_s = pickle.load(open('s_prediction.sav','rb'))

st.title("Synthesis of Design Parameters of a Square Loop based FSS using ML Approach/Technique")

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return pybase64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('./bg2.png')

st.header('For FR-4 Substrate')

st.caption('The Range for Resonant Frequency: 0 < fr < 5') 
st.caption('The Range for Lower-cutoff Frequency: 0 < fl < 5')
st.caption('The Range for Higher-cutoff Frequency: 0 < fh < 5')
col1, col2, col3 = st.columns(3)
with col1:
    h = st.selectbox('Height of the Substrate in mm (h)',(0.8,1.6,3.2))
with col2:
    fr = st.number_input("Resonant Frequency in GHz (fr)",min_value=0.01,max_value=5.00,step=0.01)
with col3:
    fl = st.number_input("Lower-cutoff Frequency in GHz (fl)",min_value=0.01,max_value=5.00,step=0.01)
with col1:
    fh = st.number_input("Higher-cutoff Frequency in GHz (fh)",min_value=0.01,max_value=5.00,step=0.01)
bw = fh - fl
fbw = bw / fr
with col2:
    g = st.selectbox('Inter Element Spacing in mm (g)',(0.25,0.375,0.5))

# creating a button for Prediction    
if st.button("Predict"):
    col1, col2 = st.columns(2)
    
    d_pred = model_d.predict([[h,fr,fl,fh,bw,fbw,g]])
    d_pred=np.round(d_pred,2)
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
    s_pred=np.round(s_pred,2)
    col2.metric(label="Track Width in mm is: ",value=s_pred)  

