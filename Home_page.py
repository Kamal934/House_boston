import streamlit as st 
import pandas as pd
import numpy as np
import warnings
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

File ='housing.csv'
name = ['CRIM', 'ZN', 'INDUS', 'CHAS','NOX','RM','AGE','DIS', 'RAD','TAX','PTRATIO','B_1000','LSTAT','Price(target)']
df= pd.read_csv(File,encoding='latin-1',delim_whitespace=True,names=name)

X = df.drop(['Price(target)'], axis = 1)
y = df['Price(target)']

st.title(' Boston House Prices App')
st.write('---')

# sidebar
st.sidebar.title("Specify Input Parameters")

custom_css="""<style>.css-ffhzg2 {
    position: absolute;
    background: black;
    color: white;
    inset: 0px;
    overflow: hidden;
}
    .css-6qob1r {
    position: relative;
    height: 100%;
    background: black;
    width: 100%;
    overflow: overlay;
}   
    css-fblp2m {
    vertical-align: middle;
    overflow: hidden;
    color: inherit;
    fill: currentcolor;
    display: inline-flex;
    -webkit-box-align: center;
    align-items: center;
    font-size: 3.25rem;
    width: 2.25rem;
    height: 3.25rem;
}   
    .css-1544g2n {
    padding: 3rem 1rem 1.5rem;
}</style>"""

st.markdown(custom_css,unsafe_allow_html=True)

# Streamlit app header
# st.title("Boston House Price Prediction")
def user_input_features():
# Collect user input for parameters using Streamlit components
    CRIM = st.sidebar.slider('CRIM',X.CRIM.min(),X.CRIM.max(),X.CRIM.mean())
    ZN = st.sidebar.slider('ZN',X.ZN.min(),X.ZN.max(),X.ZN.mean())
    INDUS = st.sidebar.slider('INDUS',X.INDUS.min(),X.INDUS.max(),X.INDUS.mean())
    CHAS = st.sidebar.selectbox('CHAS', [0, 1], index=0)    
    NOX= st.sidebar.slider('NOX',X.NOX.min(),X.NOX.max(),X.NOX.mean())
    RM=st.sidebar.selectbox('RM', [3,4,5,6,7,8], index=0)  
    AGE=st.sidebar.slider('AGE',int(X.AGE.min()),int(X.AGE.max()),int(X.AGE.mean()))
    DIS=st.sidebar.slider('DIS',X.DIS.min(),X.DIS.max(),X.DIS.mean())
    RAD=st.sidebar.slider('RAD',int(X.RAD.min()),int(X.RAD.max()),int(X.RAD.mean()))
    TAX=st.sidebar.slider('TAX',X.TAX.min(),X.TAX.max(),X.TAX.mean())
    PTRATIO=st.sidebar.slider('PTRATIO',X.PTRATIO.min(),X.PTRATIO.max(),X.PTRATIO.mean())
    B_1000=st.sidebar.slider('B_1000',X.B_1000.min(),X.B_1000.max(),X.B_1000.mean())
    LSTAT=st.sidebar.slider('LSTAT',X.LSTAT.min(),X.LSTAT.max(),X.LSTAT.mean())
    data={'CRIM':CRIM,
        'ZN':ZN,
        'INDUS':INDUS,
        'CHAS':CHAS,
        'NOX':NOX,
        'RM':RM,
        'AGE':AGE,
        'DIS':DIS,
        'RAD':RAD,
        'TAX':TAX,
        'PTRATIO':PTRATIO,
        'B_1000':B_1000,
        'LSTAT':LSTAT}
    features=pd.DataFrame(data,index=[0])
    return features
user_input=user_input_features()


# st.write(func)
reg=LinearRegression()
reg.fit(X,y)
prediction= reg.predict(user_input)

st.sidebar.button('Reset')
col1,col2=st.columns(2)
with col1:
    st.header("Prediction of Price")
    formated_pred=round(float(prediction), 2)
    formated_pred_str = f"$ {formated_pred}"
    if formated_pred <= 0:
        st.markdown(f"<h2 style='font-size: 65px; color:green;'>{' $ 0'}</h2>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h2 style='font-size: 65px; color:green;'>{formated_pred_str}</h2>", unsafe_allow_html=True)
with col2:
    st.image("house.png", width=470)
st.write("---")

# Create a KernelExplainer for the linear regression model
explainer = shap.KernelExplainer(reg.predict, X)
shap_values=explainer.shap_values(user_input)
st.header('Feature Importance')
plt.title("Feature importance based on the shap values")

# Create a summary plot of absolute SHAP values

fig, ax = plt.subplots()
# plt.figure(facecolor='black')
plt.title("Feature importance based on the shap values")

shap.summary_plot(shap_values, feature_names=X.columns, plot_type="bar",show=False,color='green')
plt.rcParams['axes.facecolor'] = 'black'  # Set the background color to black

st.pyplot(fig,bbox_inches='tight')

st.write('---')
# st.header("SHAP Values")
# st.write(shap_values)


hide_st_style="""
<style>
#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
</style>"""
st.markdown(hide_st_style,unsafe_allow_html=True)