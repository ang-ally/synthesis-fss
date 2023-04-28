import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import streamlit as st
import plotly.graph_objs as go

model_d = pickle.load(open('d_prediction.sav','rb'))
model_s = pickle.load(open('s_prediction.sav','rb'))

st.set_page_config(page_title="Square Loop FSS Synthesis")

st.header("Synthesis of Design Parameters of a Square Loop based FSS using ML Technique")

st.header('For FR-4 Substrate')

st.subheader('Incident angle 𝜃= 0°')

st.subheader('Mode of Operation: TE')

st.caption('The Range for Resonant Frequency (fr): 0 < fr < 5 GHz') 
st.caption('The Range for Lower cutoff Frequency (fl): 0 < fl < 5 GHz')
st.caption('The Range for Higher cutoff Frequency (fh): 0 < fh < 5 GHz')

h = st.selectbox('Height of the Substrate in mm (h)',(0.8,1.6,3.2))
fr = st.number_input("Resonant Frequency in GHz (fr)",min_value=0.01,max_value=5.00,step=0.01)
fl = st.number_input("Lower cutoff Frequency in GHz (fl)",min_value=0.01,max_value=5.00,step=0.01)
fh = st.number_input("Higher cutoff Frequency in GHz (fh)",min_value=0.01,max_value=5.00,step=0.01)
BW = fh - fl
FBW = BW / fr
g = st.selectbox('Inter-element Spacing in mm (g)',(0.25,0.375,0.5))

# creating a button for Prediction    
if st.button("Synthesize"):
    if fl<fr and fr<fh and fl<fh:
        col1, col2 = st.columns(2)

        d_pred = model_d.predict([[h,fr,fl,fh,BW,FBW,g]])
        d_pred=np.round(d_pred,2)
        col1.metric(label="Track Length in mm is : ",value=d_pred)

        df=pd.read_excel('final fr4 ds.xlsx')
        X=df[['h','fr','fl','fh','BW','FBW','g']]
        Y=df[['s']]
        scale_in=RobustScaler()
        scale_out=RobustScaler()
        x=scale_in.fit_transform(X)
        y=scale_out.fit_transform(Y)
        prediction=scale_in.transform([[h,fr,fl,fh,BW,FBW,g]])
        s_pred=model_s.predict(prediction)
        s_pred=s_pred.reshape(-1,1)
        s_pred=scale_out.inverse_transform(s_pred)
        s_pred=s_pred.reshape(1,-1) 
        s_pred=np.round(s_pred,2)
        col2.metric(label="Track Width in mm is: ",value=s_pred) 
        d = int(d_pred[0])
        s = int(s_pred[0])
        g = int(g)
        p = d + g
        a = d - s
        b = p + s
        c = (2*d) + g
        e = c - s

        # Create a line plot using plotly
        fig = go.Figure()

        #loop 1 
        l1_x1 = [0, d, d, 0, 0]
        l1_y1 = [0, 0, d, d, 0]

        l1_x2 = [s, a, a, s, s]
        l1_y2 = [s, s, a, a, s]

        fig.add_trace(go.Scatter(x=l1_x1, y=l1_y1,
                            mode='lines',showlegend=False,name='sqr1',line_color='gray'))

        fig.add_trace(go.Scatter(x=l1_x2, y=l1_y2,
                            mode='lines',showlegend=False,name='sqr2',line_color='gray'))

        fig.add_trace(go.Scatter(x=np.concatenate([l1_x1, l1_x2[::-1]]), y=np.concatenate([l1_y1, l1_y2[::-1]]), 
                                 fill='toself', fillcolor='gray', line_color='rgba(0,0,0,0)', 
                                 showlegend=False,hoverinfo='skip'))

        #loop 2 
        l2_x1 = [p, c, c, p, p]
        l2_y1 = [0, 0, d, d, 0]

        l2_x2 = [b, e, e, b, b]
        l2_y2 = [s, s, a, a, s]

        fig.add_trace(go.Scatter(x=l2_x1, y=l2_y1,
                            mode='lines',showlegend=False,name='sqr1',line_color='gray'))

        fig.add_trace(go.Scatter(x=l2_x2, y=l2_y2,
                            mode='lines',showlegend=False,name='sqr2',line_color='gray'))

        fig.add_trace(go.Scatter(x=np.concatenate([l2_x1, l2_x2[::-1]]), y=np.concatenate([l2_y1, l2_y2[::-1]]), 
                                 fill='toself', fillcolor='gray', line_color='rgba(0,0,0,0)', 
                                 showlegend=False,hoverinfo='skip'))

        #loop 3 
        l3_x1 = [p, c, c, p, p]
        l3_y1 = [p, p, c, c, p]

        l3_x2 = [b, e, e, b, b]
        l3_y2 = [b, b, e, e, b]

        fig.add_trace(go.Scatter(x=l3_x1, y=l3_y1,
                            mode='lines',showlegend=False,name='sqr1',line_color='gray'))

        fig.add_trace(go.Scatter(x=l3_x2, y=l3_y2,
                            mode='lines',showlegend=False,name='sqr2',line_color='gray'))

        fig.add_trace(go.Scatter(x=np.concatenate([l3_x1, l3_x2[::-1]]), y=np.concatenate([l3_y1, l3_y2[::-1]]), 
                                 fill='toself', fillcolor='gray', line_color='rgba(0,0,0,0)', 
                                 showlegend=False,hoverinfo='skip'))

        #loop 4 
        l4_x1 = [0, d, d, 0, 0]
        l4_y1 = [p, p, c, c, p]

        l4_x2 = [s, a, a, s, s]
        l4_y2 = [b, b, e, e, b]

        fig.add_trace(go.Scatter(x=l4_x1, y=l4_y1,
                            mode='lines',showlegend=False,name='sqr1',line_color='gray'))

        fig.add_trace(go.Scatter(x=l4_x2, y=l4_y2,
                            mode='lines',showlegend=False,name='sqr2',line_color='gray'))

        fig.add_trace(go.Scatter(x=np.concatenate([l4_x1, l4_x2[::-1]]), y=np.concatenate([l4_y1, l4_y2[::-1]]), 
                                 fill='toself', fillcolor='gray', line_color='rgba(0,0,0,0)', 
                                 showlegend=False,hoverinfo='skip'))

        # Add an annotation to the chart 
        #for p
        # add a line shape
        fig.add_shape(
            type='line',
            x0=0, y0=c, x1=0, y1=c+1,
            line=dict(color='gray', width=1, dash='solid'),
        )
        # add a line shape
        fig.add_shape(
            type='line',
            x0=p, y0=c, x1=p, y1=c+1,
            line=dict(color='gray', width=1, dash='solid'),
        )

        fig.add_annotation(x=0 , y=c, ax=p, ay=c,
                           showarrow=True,xref='x1',yref='y1',
                            axref='x1', ayref='y1',
                           arrowcolor='black', arrowwidth=2,arrowside='start+end', arrowhead=None, yshift=10)
        fig.add_annotation(x=(p/2) , y=c, ax=0, ay=c,yshift=20,
                           showarrow=False,text=f'Periodicity (p): {p:.2f} mm',    font=dict(
                family="Arial",
                size=14,
                color="black"
            ))

        #for g
        # add a line shape
        fig.add_shape(
            type='line',
            x0=c, y0=p, x1=c+1, y1=p,
            line=dict(color='gray', width=1, dash='solid'),
        )
        # add a line shape
        fig.add_shape(
            type='line',
            x0=c, y0=d, x1=c+1, y1=d,
            line=dict(color='gray', width=1, dash='solid'),
        )
        fig.add_annotation(x=c , y=p, ax=c, ay=b+1,
                           showarrow=True,xref='x1',yref='y1',
                            axref='x1', ayref='y1',
                           arrowcolor='black', arrowwidth=1.5, arrowhead=1,xshift=12)
        fig.add_annotation(x=c , y=d, ax=c, ay=a-1,
                           showarrow=True,xref='x1',yref='y1',
                            axref='x1', ayref='y1',
                           arrowcolor='black', arrowwidth=1.5,arrowhead=1,xshift=12)                   
        fig.add_annotation(x=c+1.5+g , y=d, ax=c+1.5+g, ay=0,textangle=-90,xshift=20+g,
                           showarrow=False,text=f'Inter-element Spacing (g): {g:.3f} mm',  font=dict(
                family="Arial",
                size=14,
                color="black"
            ))

        #for s
        fig.add_annotation(x=s , y=d/2, ax=0, ay=d/2,standoff=0,
                           showarrow=True,xref='x1',yref='y1',
                            axref='x1', ayref='y1',
                           arrowcolor='black', arrowwidth=2,arrowside='start+end', arrowhead=None)
        fig.add_annotation(x=0 , y=d/2, ax=0, ay=0,xshift=-15,textangle=-90,
                           showarrow=False,text=f'Track width (s): {s:.2f} mm',  font=dict(
                family="Arial",
                size=14,
                color="black"
            ))

        #for d
        # add a line shape
        fig.add_shape(
            type='line',
            x0=0, y0=0, x1=0, y1=-1,
            line=dict(color='gray', width=1, dash='solid'),
        )
        # add a line shape
        fig.add_shape(
            type='line',
            x0=d, y0=0, x1=d, y1=-1,
            line=dict(color='gray', width=1, dash='solid'),
        )

        fig.add_annotation(x=d , y=0, ax=0, ay=0,
                           showarrow=True,xref='x1',yref='y1',
                            axref='x1', ayref='y1',
                           arrowcolor='black', arrowwidth=2,arrowside='start+end', arrowhead=None, yshift=-10)
        fig.add_annotation(x=d/2 , y=0, ax=0, ay=0,yshift=-20,
                           showarrow=False,text=f'Track length (d): {d:.2f} mm',    font=dict(
                family="Arial",
                size=14,
                color="black"
            ))

        # Set chart title and axes labels
        fig.update_layout(title="Square Loop FSS")

        fig.update_xaxes(tickmode="linear", tick0=0, dtick=g, rangemode="normal")
        fig.update_yaxes(tickmode="linear", tick0=0, dtick=g, rangemode="normal")

        # Remove the x and y axes and their labels and tick marks
        fig.update_xaxes(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
        )
        fig.update_yaxes(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
        )

        fig.update_layout(
            width=750,
            height=750
        )

        # Display the plot using Streamlit
        st.plotly_chart(fig)
    else:
        if fl>=fr and fr>=fh and fl>=fh:
            st.error('The Resonant frequency(fr) should be lie between the Lower cutoff frequency(fl) and the Higher cutoff frequency(fh)  (i.e) fl < fr < fh ', icon="🚨")
        else:
            with st.container():
                if fl>=fr:
                    st.error('The Resonant frequency(fr) should be higher than the Lower cutoff frequency(fl) ', icon="🚨")
                if fr>=fh:
                    st.error('The Resonant frequency(fr) should be lower than the Higher cutoff frequency(fh) ', icon="🚨")
                if fl>=fh:
                    st.error('The Lower cutoff frequency(fl) should be lower than the Higher cutoff frequency(fh) ', icon="🚨")
