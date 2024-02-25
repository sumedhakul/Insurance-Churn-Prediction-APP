#Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle
import base64


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib.pyplot as plt
import geopandas as gpd
# folium for interactive maps
import folium


####################


# set background
@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64('edit.jpg')
page_bg_img = f"""
    <style> 
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{img}");
        background-size: cover;
    }}
    [data-testid="stHeader"] {{
        background-color: rgba(0, 0, 0, 0)
    }}
    [data-testid="stToolbar"] {{
        right: 2rem;
    }}
    
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
#########################################


#load the model from disk
import joblib
model = joblib.load(r"new_model.sav")





#Import python scripts
from pre import preprocess

def main():
    #Setting Application title
    st.title('Insurance Customer Churn Prediction App')


    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("E:\\Sumedha(important)\\4th Year 2nd Sem\\Research\\Kaggle dataset\\DEPLOY\\images.jpeg");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


      #Setting Application description
    st.markdown("""
     :dart:  This Streamlit app is made to predict customer churn in a ficitional Insurance use case.
    The application is functional for both online prediction and batch data prediction.
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    #Setting Application sidebar default
    image = Image.open('E:\\Sumedha(important)\\4th Year 2nd Sem\\Research\\Kaggle dataset\\DEPLOY\\small.png')
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?", ("Online", "Upload CSV"))
    st.sidebar.info('Predict Customer Churn')
    st.sidebar.image(image)

    if add_selectbox == "Online":
        st.info("Input data below")
       
    
    #Based on our optimal features selection
    
        curr_ann_amt = st.number_input('The curr_ann_amtr',min_value=0, max_value=2000, value=0)
        days_tenure = st.number_input('The days_tenure', min_value=0, max_value=10000, value=0)
        age_in_years = st.number_input('The age_in_yearsr',min_value=0, max_value=100, value=0)
        income = st.number_input('The income',min_value=0, max_value=150000, value=0)
        
        st.subheader("Demographic data")
        has_children=st.number_input('has_children: Yes=1 ,No = 0',min_value=0, max_value=1, value=0,step = 1)
        length_of_residence=st.number_input('length_of_residence:',min_value=0, max_value=15, value=0,step = 1)
        marital_status = st.number_input('Marital Status: Married=1 ,Single = 0',min_value=0, max_value=1, value=0,step = 1)
        home_owner=st.number_input('home_owner: Yes=1 ,No = 0',min_value=0, max_value=1, value=0,step = 1)
        college_degree=st.number_input('college_degree: Yes=1 ,No = 0',min_value=0, max_value=1, value=0,step = 1)
        good_credit=st.number_input('good_credit: Yes=1 ,No = 0',min_value=0, max_value=1, value=0,step = 1)
        

        data ={
                'curr_ann_amt':curr_ann_amt, 
                'days_tenure':days_tenure, 
                'age_in_years':age_in_years, 
                'income':income, 
                'has_children':has_children,
                'length_of_residence':length_of_residence, 
                'marital_status':marital_status, 
                'home_owner':home_owner, 
                'college_degree':college_degree,
                'good_credit':good_credit
        
        
                }
        features_df = pd.DataFrame.from_dict([data])
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.write('Overview of input is shown below')
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.dataframe(features_df)
        #Preprocess inputs
        preprocess_df = preprocess(features_df, 'Online')

        prediction = model.predict(preprocess_df)

        if st.button('Predict'):
            if has_children == 1:
                st.warning('Yes, the customer will terminate the service.')
            else:
                st.success('No, the customer is happy with Insurance Services.')


    else:
        st.subheader("Dataset upload")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            #Get overview of data
            st.write(data.head())
            st.markdown("<h3></h3>", unsafe_allow_html=True)
            #Preprocess inputs
            preprocess_df = preprocess(data, "Batch")
            if st.button('Predict'):
                #Get batch prediction
                prediction = model.predict(preprocess_df)
                prediction_df = pd.DataFrame(prediction, columns=["prediction"])
                prediction_df = prediction_df.replace({1:'Yes, the customer will terminate the service.',
                                                    0:'No, the customer is happy with The Insurance Services.'})

                st.markdown("<h3></h3>", unsafe_allow_html=True)
                st.subheader('Prediction')
                st.write(prediction_df)

if __name__ == '__main__':
        main()