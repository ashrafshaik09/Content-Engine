# Content Engine

## Overview

The Content Engine is a system that analyzes and compares multiple PDF documents, specifically identifying and highlighting their differences. It uses Retrieval Augmented Generation (RAG) techniques to retrieve, assess, and generate insights from the documents.

## Features

- **Document Parsing and Processing:** Extracts text and structure from PDF documents.
- **Embedding Generation:** Uses a local embedding model to create embeddings for document content.
- **Vector Store:** Stores and queries embeddings using Chroma.
- **Local LLM:** Integrates a local instance of the Llama model for generating insights.
- **Chatbot Interface:** Uses Streamlit to facilitate user interaction and display insights.

## Setup

### Prerequisites

- Install Python 3.8 or higher
- Install CMake: [Download CMake](https://cmake.org/download/)
- Install Visual Studio Build Tools: [Download VS Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

### Installation

1. **Clone the Repository**

    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2. **Create and Activate Virtual Environment**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    pip install llama-cpp-python langchain-community langchain-huggingface
    pip install --upgrade llama-cpp-python pydantic
    ```

4. **Download and Setup the Model**

    - Download the Llama model from [HuggingFace](https://huggingface.co/)
    - Place the model file (`Meta-Llama-3-8B-Instruct.Q2_K.gguf`) in the `models` directory.

5. **Run the Application**

    ```bash
    streamlit run app.py
    ```

### Usage

1. **Upload PDF Documents:**

    - Use the sidebar to upload your PDF documents.
    - Click on 'Process' to parse and store the documents.

2. **Ask Questions:**

    - Enter your query in the text input box and click 'Ask'.
    - The system will retrieve relevant information from the documents and generate a response.

### Project Structure

- `app.py`: Streamlit application to handle user interface and interactions.
- `main.py`: Main script to run the Streamlit app.
- `rag_system.py`: Core logic for parsing documents, generating embeddings, storing vectors, and querying.
- `htmlTemplates.py`: HTML and CSS templates for the Streamlit interface.
- `requirements.txt`: List of dependencies.

### Example Queries

1. **What are the risk factors associated with Google and Tesla?**
2. **What is the total revenue for Google Search?**
3. **What are the differences in the business of Tesla and Uber?**

### Troubleshooting

- **No pages loaded:** Ensure the PDF files are correctly formatted and accessible.
- **Unexpected response format:** Check the model integration and ensure the correct model is being used.

### Contribution

Feel free to fork this repository and contribute by submitting a pull request.

### License

This project is licensed under the MIT License.

---

Feel free to reach out if you have any questions or need further assistance.
