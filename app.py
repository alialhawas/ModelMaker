from operator import index
from streamlit_chat import message
import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
from ydata_profiling import ProfileReport
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os 


file_path = "./dataset_noNA.csv"

if os.path.exists(file_path): 
    df = pd.read_csv(file_path, index_col=None)

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("myAutoML")
    choice = st.radio("Navigation", ["About", "Upload","Profiling","Modelling", "Download", "talk to ahmed"])
    st.info("This project application helps you build and explore your data.")


if choice == "About":
    st.title("About")
    st.write("This is a project that helps you build and explore your data. You can upload your dataset, explore it, and build a model. You can also download the model for future use.")
    st.write("This project is built using AI and ML libraries such as PyCaret, to help you imporve your Ai anmd ML models")
    st.write("built by ali Alhwas")


if choice == "talk to Ahmed":
    st.title("Talk to Ahmed")



if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        st.dataframe(df)

if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    mydf = pd.read_csv('./dataset.csv', index_col=None)
    profile_df = ProfileReport(mydf, title='My Data')
    st_profile_report(profile_df)

if choice == "Modelling": 
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'): 
        setup(df, target=chosen_target)
        setup_df = pull()
        st.dataframe(setup_df) 
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')
        with open('best_model.pkl', 'rb') as f: 
            st.download_button('Download Model', f, file_name="best_model.pkl")

if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")