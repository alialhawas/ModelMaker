import streamlit as st
from streamlit_chat import message
from streamlit_pandas_profiling import st_profile_report 


import plotly.express as px
from pycaret.regression import setup as reg_setup, compare_models as reg_compare_models, pull as reg_pull, save_model as reg_save_model, load_model as reg_load_model
from pycaret.classification import setup as class_setup, compare_models as class_compare_models, pull as class_pull , save_model as class_save_model, load_model as class_load_model
from ydata_profiling import ProfileReport



import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import os

from pandasai import SmartDataframe
from langchain_groq import ChatGroq


from config import groq_api_key
from utils.index import setup_cleaning_chatbot, clening_data, custom_cleaning, setup_modling_chatbot

file_path = "./dataset_noNA.csv"

clening_action = ['Fix Type', 'Drop Column', 'Fill Missing with Avg', '']

if 'df' not in st.session_state:
    st.session_state['df'] = pd.read_csv(file_path, index_col=None) 



if os.path.exists(file_path): 
    df = pd.read_csv(file_path, index_col=None) # TODO: better handle the upload file

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("my AI ML")
    choice = st.radio("Navigation", ["About", "Upload","Analysis",
                                     "Data Preprocessing","Modelling", "Talk to AI"])
    
    st.info("This project application helps you build and explore your data.")


if choice == "About":

    st.title("About")
    st.write("This is a project that helps you build and explore your data. You can upload your dataset, explore it, and build a model. You can also download the model for future use.")
    st.write("This project is built using AI and ML libraries, to help you imporve your AI and ML models")



if choice == "Upload":
    st.title("Upload Your Dataset")

    # st.write("Upload your dataset to get started with the analysis.")
    # business_value = st.text_input("Query:", placeholder="Describe and the desired business value that will be extracted from it.", key='input')
    # todo add  business_value to the bot to help the user to get the best value from the data

    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        # st.dataframe(df)
        st.session_state['df'] = df

if choice == "Analysis": 

    st.title("Exploratory Data Analysis (ETA)")

    profile = ProfileReport(  st.session_state['df'], title='My Data')
   
    st_profile_report(profile)


if choice == "Data Preprocessing":

    # """
    # To Have An Asstent To Help You Clean The Data:
    # - First give a summrize of the proplems in the data
    # - Then give the user the option to clean the data with the defult cleaning or the custom cleaning
    # - The user can choose to drop the columns with high missing values ..ect
    # - Create a model to predict the missing values to not loss the data
    # - Note: what the the defult cleaning will convert the columns to their real types
    # """

    st.title("Data Preprocessing")

    setup_cleaning_chatbot()

    if st.button('Defult Cleaning'): 
       st.session_state['df'] = clening_data(df= st.session_state['df']) # TODO: find a better way than mutation.
       st.success('Your Data Has Been Cleaned!', icon="✅")


    if st.button('Custom Cleaning'): 

        chosen_column = st.selectbox('Choose the Target Column', df.columns)
        clening_action = st.selectbox('Choose what to do with the column', clening_action)
        
        if st.button("excute"):
            st.session_state['df'] = custom_cleaning(df= st.session_state['df']) # TODO : two layers of button does't work   
            st.success('The action taken is succesfull', icon="✅")

            # try : 
            #     st.session_state['df'] = custom_cleaning(df= st.session_state['df'])
            #     st.success('The action taken is succesfull', icon="✅")
            # except Exception as e :
            #     message = f" an error has occed with the action takan to  the selected column {chosen_column} with a this message {e} ."
            #     st.error(message, icon="❌")
            


if choice == "Modelling": 

    setup_modling_chatbot(st.session_state['df'])

    chosen_target = st.selectbox('Choose the Target Column', st.session_state['df'].columns)

    if st.button('Run Modelling'): 

        if chosen_target in  st.session_state['df'].select_dtypes(include=['int', 'float']).columns:

            reg_setup(st.session_state['df'], target=chosen_target)
            setup_df = reg_pull()

            st.dataframe(setup_df) 

            best_model = reg_compare_models()

            compare_df = reg_pull()

            st.dataframe(compare_df)

            reg_save_model(best_model, 'best_model')
            with open('best_model.pkl', 'rb') as f: # TODO find a batter way to save the model  
                st.download_button('Download Model', f, file_name="best_model.pkl") 


        elif chosen_target in st.session_state['df'].select_dtypes(include=['object', 'category']).columns:
            class_setup(st.session_state['df'], target=chosen_target)
            setup_df = class_pull()

            st.dataframe(setup_df) 

            best_model = class_compare_models()

            compare_df = class_pull()

            st.dataframe(compare_df)

            class_save_model(best_model, 'best_model')
            with open('best_model.pkl', 'rb') as f: # TODO find a batter way to save the model  
                st.download_button('Download Model', f, file_name="best_model.pkl") 
        
        else:
            st.error("The target column should be either an integer, float, object, or category type", icon="❌")


if choice == "Talk to AI":

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me anything about your data 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]

    st.title("Talk To Ahmad")
    st.write("Ahmad Can Help You With Your Questions. Please Type Your Question Below.")
    

    l_llm = chat = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-8b-8192") # Model Name: llama3-8b-8192 | mixtral-8x7b-32768

    df_ai = SmartDataframe(st.session_state['df'], config={"llm": l_llm})

    # container for the chat history
    response_container = st.container()

    # container for the user's text input
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
