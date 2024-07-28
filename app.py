from operator import index
from streamlit_chat import message
import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
from ydata_profiling import ProfileReport
import pandas as pd
from streamlit_pandas_profiling import st_profile_report 

import streamlit as st 
from streamlit_chat import message

import pandas as pd,matplotlib.pyplot as plt, seaborn as sns
import os
from pandasai import SmartDataframe
from langchain_groq import ChatGroq
from config import groq_api_key

"""
things to do:
    - add a section to drop unused columns
    - add a section to drop rows with missing values
    - handle high cardinality columns
    - handle high missing values columns in two ways: 1- drop the column 2-  create a model to predict the missing values to not loss the data
    - handel high correlation columns 
"""

file_path = "./dataset_noNA.csv"

if os.path.exists(file_path): 
    df = pd.read_csv(file_path, index_col=None) # TODO: better handle the upload file

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("myAutoML")
    choice = st.radio("Navigation", ["About", "Upload","Analysis","Modelling", "Download", "Talk to AI"])
    st.info("This project application helps you build and explore your data.")


if choice == "About":

    st.title("About")
    st.write("This is a project that helps you build and explore your data. You can upload your dataset, explore it, and build a model. You can also download the model for future use.")
    st.write("This project is built using AI and ML libraries such as PyCaret, to help you imporve your Ai anmd ML models")
    st.write("built by ali Alhwas")


if choice == "Talk to AI":


    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me anything about your data 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]

    st.title("Talk To Ahmed")
    st.write(" Ahmed Can Help You With Your Questions. Please Type Your Question Below.")


    
    df = pd.read_csv("./dataset.csv")

    l_llm = chat = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-8b-8192") # Model Name: llama3-8b-8192 | mixtral-8x7b-32768

    df_ai = SmartDataframe(df, config={"llm": l_llm})


    #container for the chat history
    response_container = st.container()

    #container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            
            user_input = st.text_input("Query:", placeholder="Talk to your data here (:", key='input')
            submit_button = st.form_submit_button(label='Send')
            
        if submit_button and user_input:
            output = df_ai.chat(user_input)
            
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")



    # add a sidebar to customize the chatbot
    # temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    # top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    # max_length = st.sidebar.slider('max_length', min_value=32, max_value=128, value=120, step=8) # TODO: add these to the sidebar to customize the chatbot



if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        st.dataframe(df)
        st.session_state.df = df

if choice == "Analysis": 

    st.title("Exploratory Data Analysis (ETA)")

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
        with open('best_model.pkl', 'rb') as f: # TODO find a batter way to save the model  
            st.download_button('Download Model', f, file_name="best_model.pkl") 

if choice == "Download": 

    with open('best_model.pkl', 'rb') as f:# TODO find a batter way to save the model  
        st.download_button('Download Model', f, file_name="best_model.pkl")