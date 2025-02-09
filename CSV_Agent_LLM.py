# Required imports
import streamlit as st
import pandas as pd
import os
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="CSV Analysis Chatbot",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Title and description
st.title("CSV/Excel Analysis Chatbot")
st.write("Upload your file and start chatting with your data!")

# Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")

# For debugging - show first 10 chars of API key (remove in production)
if api_key:
    st.write(f"API Key found: {api_key[:10]}...")
else:
    st.error("API key not found!")
    st.stop()

# File upload
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=['csv', 'xlsx'])

if uploaded_file is not None:
    # Read the file
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Add a separator between preview and chat
        st.write("Data Preview:")
        st.dataframe(df.head())
        st.markdown("---")
        st.write("💬 Chat with your data:")

        # Create the agent with explicit API key
        llm = ChatOpenAI(
            api_key=api_key,  # Explicitly pass the API key here
            model="gpt-4",
            temperature=0
        )
        
        agent = create_pandas_dataframe_agent(
            llm,  # Use the configured LLM
            df,
            verbose=True,
            allow_dangerous_code=True
        )

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask anything about your data..."):
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner('Analyzing...'):
                    try:
                        response = agent.invoke(prompt)
                        st.write(response["output"])
                        # Add AI response to history
                        st.session_state.messages.append({"role": "assistant", "content": response["output"]})
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
else:
    st.info("Please upload a file to begin analysis.")
    st.markdown("""
    ### Example questions you can ask:
    - What is the total number of rows and columns in this dataset?
    - What are the main statistical measures for column X?
    - Can you show me the top 5 values in column Y?
    - Is there any correlation between column A and B?
    - What are the unique values in column Z?
    - Generate a summary of the main insights from this data
    - What are the minimum and maximum values in each column?
    - Are there any missing values in the dataset?
    """)

# Add footer
st.markdown("---")
st.markdown("Built with ❤️ using Langchain, OpenAI, and Streamlit") 