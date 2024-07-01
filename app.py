import streamlit as st
from rag_system import RAGSystem
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template

# Initialize the RAG system
rag_system = RAGSystem()

def handle_question(question):
    response = rag_system.answer_query(question)
    st.session_state.chat_history.append({"role": "user", "content": question})
    st.session_state.chat_history.append({"role": "assistant", "content": response})

def main():
    load_dotenv()
    st.set_page_config(page_title="Content Engine", page_icon="📖")
    st.write(css, unsafe_allow_html=True)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Content Engine")
    question = st.text_input("Ask question from your document:")
    if st.button("Ask"):
        handle_question(question)
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.write(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader("Upload your PDF here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                rag_system.process_documents(docs)

if __name__ == "__main__":
    main()
