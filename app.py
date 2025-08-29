from dotenv import load_dotenv
import os
from langchain_openai import AzureChatOpenAI
import openai
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import streamlit as st




load_dotenv()

st.set_page_config(layout="wide", page_title="Document Chat NCERT", page_icon='Fav_icon.png')


VALID_CREDENTIALS = {
    "sridhar": "Excel@123",
    "abhishek": "Excel@123",
    "shwetha": "Excel@123",
    "anu": "Excel@123",
    "kulkarni@excelsoftcorp.com": "Excel@123",
    "adarsh@excelsoftcorp.com": "Excel@123",
    "rakesh.sharma@excelsoftcorp.com": "Excel@123",
    "Guest1 ": "P@ssword",
    "Guest2 ": "P@ssword",
    "Guest3 ": "P@ssword",
    "Guest4 ": "P@ssword",
    "Guest5 ": "P@ssword",
    "trialuseracc@excelsoftcorp.com": "acc@123",
    "trailusertquk@excelsoftcorp.com": "tquk@1234",
    "trailuseraqa@excelsoftcorp.com": "aqa@12345",
    "trailhooder@excelsoftcorp.com": "Excel@1234",
    # Add more users as needed
}

def login():
    
    custom_css = """
    <style>
    body {
        background-color: #f0f2f6;
        font-family: Arial, sans-serif;
        color: #333;
    }

    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }

    .stButton>button:hover {
        background-color: #45a049;
    }

    .stTextInput>div>div>input[type="text"],
    .stTextInput>div>div>input[type="password"] {
        background-color: #f8f9fa;
        border: 1px solid #ccc;
        border-radius: 4px;
        padding: 8px 12px;
    }

    .stTextInput>div>div>input[type="text"]:focus,
    .stTextInput>div>div>input[type="password"]:focus {
        outline: none;
        border-color: #4CAF50;
        box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.2);
    }

    .st-emotion-cache-uf99v8 {
        background: linear-gradient(135deg, #ff00ff, #00ffff);
    }

    .st-emotion-cache-pb6fr7 {
        width: 407px;
    }

    .st-emotion-cache-gh2jqd {
        max-width: 31rem;
    }
    .st-emotion-cache-z5fcl4 {
    width: 40%;
    }

    </style>
    """

    st.markdown(custom_css, unsafe_allow_html=True)

    st.title("Excelsoft - DocuChat")
    with st.form(key='login_form'):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if username in VALID_CREDENTIALS and password == VALID_CREDENTIALS[username]:
                return True
            else:
                st.error("Invalid username or password. Please try again.")
                return False

def main():
    
    # Check if the user is already logged in
    if "is_logged_in" not in st.session_state:
        st.session_state.is_logged_in = False

    if not st.session_state.is_logged_in:
        # If the user is not logged in, display the login page
        if login():
            st.session_state.is_logged_in = True
            st.rerun()
    else:
        # UI layout after successful login
        st.empty()
        logo_path = "excellogo2.png"
        st.image(logo_path, use_container_width=True)
        # Add other components of your main page here
        # Load environment variables
        load_dotenv()


        # Function to process uploaded PDF file
        def process_file(uploaded_file):
            if uploaded_file is not None:
                # Save the uploaded file to a temporary location
                with open("temp2.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())

                try:
                    # Process the temporary file
                    embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'})

                    loader = PyPDFLoader("temp2.pdf")
                    pages = loader.load_and_split()
                    db = FAISS.from_documents(documents=pages, embedding=embeddings)
                    db.save_local("./dbs/documentation/faiss_index")
                    st.success("File processed successfully!")
                    st.session_state.messages = []
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

                # Remove the temporary file
                os.remove("temp2.pdf")

        # Function to ask a question and display the answer with chat history
        def ask_question(qa, question, chat_history):
            result = qa({"question": question, "chat_history": chat_history})

            # Display chat history in a user-friendly format
            for i, (q, a) in enumerate(chat_history):
                if q.startswith("You"):
                    st.markdown(f"""
                                <div style='background-color: #f1f0f0; padding: 10px; border-radius: 10px;
                                margin: 10px 0; display: inline-block; max-width: 70%; text-align: left; float: right;'>
                                <span style='color: #555555;float:right'>{a}</span>
                                </div>
                                """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                                <div style='background-color: #d6e0f0; padding: 10px; border-radius: 10px;
                                margin: 10px 0; display: inline-block; max-width: 70%; text-align: left; float: left;'>
                                <span style='margin-left: 10px;float:left'>{a}</span>
                                </div>
                                """, unsafe_allow_html=True)

            st.markdown(f"""
                        <div style='background-color: white; padding: 10px; border-radius: 10px;
                        margin: 10px 0; display: inline-block; max-width: 70%; text-align: left; float: right;'>
                        <span style='color: #555555;float:right'>{question}</span>
                        </div>
                        """, unsafe_allow_html=True)
            st.markdown(f"""
                        <div style='background-color: blue; padding: 10px; border-radius: 10px;
                        margin: 10px 0; display: inline-block; max-width: 70%; text-align: left; float: left;'>
                        <span style='margin-left: 10px;float:left'>{result["answer"]}</span>
                        </div>
                        """, unsafe_allow_html=True)

            st.session_state.messages.append(("You", question))
            st.session_state.messages.append(("", result["answer"]))

        # Add custom CSS to hide Streamlit menu and footer
        st.markdown(
            """
            <style>
            
            .st-emotion-cache-9tg1hl
            {
              padding: 3rem 1rem 1rem !important;
              position:static !important;
            }

            .st-emotion-cache-7ym5gk{margin-left: 1rem !important;}
            </style>
            
            
            """,
            unsafe_allow_html=True
        )

        # Left side: Display Book Details
        st.sidebar.title("Book Details :")

        # Display Book Name
        st.sidebar.write("Book Name: **ClassVII_science_full**")

        # Display Author Name
        st.sidebar.write("Author: **NCERT**")

        # Display Cover Image
        cover_path = "./cover2.jpg"
        st.sidebar.image(cover_path, use_container_width=True)

        #########################

        # Left side: File upload and processing
        st.sidebar.title("Upload & Process")
        uploaded_file = st.sidebar.file_uploader("Upload PDF File", type=["pdf"])

        if uploaded_file is not None and st.sidebar.button("Process File"):
            process_file(uploaded_file)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        question = st.chat_input("Say something")
        if question:
            st.spinner("Loading model...")


            try:
    # Initialize embeddings, vector store, retriever, and LLM

    # 1. Initialize local Hugging Face embeddings
                embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # 2. Load the local vector store you created
                vector_store = FAISS.load_local("./dbs/documentation/faiss_index", embeddings,allow_dangerous_deserialization=True)
                retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    # 3. Initialize the local Hugging Face LLM
    # Use "google/flan-t5-base" for less RAM usage if needed
                model_id = "google/flan-t5-large"
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    # Create a transformers pipeline
                pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.7,
        torch_dtype=torch.bfloat16,
        device_map="auto"  # Automatically uses GPU if available
    )

    # Wrap the pipeline in the LangChain LLM
                llm = HuggingFacePipeline(pipeline=pipe)

                chat_history = st.session_state.messages
                QUESTION_PROMPT = PromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

                Follow Up Input: {question}
                Standalone question:""")

    # Initialize and configure ConversationalRetrievalChain
                qa = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=retriever,
                    condense_question_prompt=QUESTION_PROMPT,
                    return_source_documents=True,
                    verbose=False
                )

    # Ask the question and update chat history
                ask_question(qa, question, st.session_state.messages)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
