import streamlit as st
from dotenv import load_dotenv


def main():
    load_dotenv()
    st.set_page_config(page_title="Chatbot", page_icon=":robot_face:")
    
    st.header("Chatbot :books:")
    st.text_input("Enter your message:")

    st.sidebar.subheader("Your documents")
    st.sidebar.file_uploader("Upload your PDF's here")
    st.sidebar.button("Process")

if __name__ == "__main__":
    main()