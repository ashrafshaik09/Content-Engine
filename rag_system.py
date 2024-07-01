from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
import os
import tempfile

class RAGSystem:
    def __init__(self, db_path="chroma", model_path="models/Meta-Llama-3-8B-Instruct.Q2_K.gguf") -> None:
        self.db_path = db_path
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.model_path = model_path
        self.prompt_template = """
                                You are an AI assistant. Use the following context to answer the question concisely and accurately. If you don't know the answer, just say that you are unsure. Do not attempt to make up an answer.
                                Context:
                                {context}
                                Question: {question}
                                Answer:
                                """
        self._initialize_vectorDB()
        self._initialize_llm()

    def _initialize_llm(self):
        self.model = LlamaCpp(model_path=self.model_path, n_ctx=2048, temperature=0)

    def process_documents(self, docs):
        with tempfile.TemporaryDirectory() as temp_dir:
            for doc in docs:
                doc_path = os.path.join(temp_dir, doc.name)
                with open(doc_path, "wb") as f:
                    f.write(doc.getbuffer())
                self._process_single_document(doc_path)

    def _process_single_document(self, doc_path):
        loader = PyPDFLoader(doc_path)
        pages = loader.load()
        if not pages:
            print("No pages loaded. Check your PDF.")
            return
        chunks = self._document_splitter(pages)
        if not chunks:
            print("No chunks created from documents. Please check the splitting process.")
            return
        chunks = self._get_chunk_ids(chunks)
        present_in_db = self.vectordb.get()
        ids_in_db = set(present_in_db["ids"])  # Convert to set for faster lookup
        print(f"Number of existing ids in db: {len(ids_in_db)}")
        chunks_to_add = [i for i in chunks if i.metadata.get("chunk_id") not in ids_in_db]
        if len(chunks_to_add) > 0:
            self.vectordb.add_documents(chunks_to_add, ids=[i.metadata["chunk_id"] for i in chunks_to_add])
            print(f"added to db: {len(chunks_to_add)} records")
            self.vectordb.persist()
        else:
            print("No records to add")


    def _get_chunk_ids(self, chunks):
        prev_page_id = None
        for i in chunks:
            src = i.metadata.get("source")
            page = i.metadata.get("page")
            curr_page_id = f"{src}_{page}"
            if curr_page_id == prev_page_id:
                curr_chunk_index += 1
            else:
                curr_chunk_index = 0
            curr_chunk_id = f"{curr_page_id}_{curr_chunk_index}"
            prev_page_id = curr_page_id
            i.metadata["chunk_id"] = curr_chunk_id
        return chunks

    def _retrieve_context_from_query(self, query_text):
        context = self.vectordb.similarity_search_with_score(query_text, k=1)
        return context

    def _get_prompt(self, query_text):
        context = self._retrieve_context_from_query(query_text)
        print(f" ***** CONTEXT ******{context} \n")
        if not context:
            return "No relevant context found for the query."
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in context])
        prompt_template = ChatPromptTemplate.from_template(self.prompt_template)
        prompt = prompt_template.format(context=context_text, question=query_text)
        return prompt

    def answer_query(self, query_text):
        prompt = self._get_prompt(query_text)
        if "No relevant context found" in prompt:
            return prompt
        response = self.model(prompt)
        print("Response from model:", response)  # Inspecting the response structure
        
        try:
            if isinstance(response, list) and len(response) > 0:
                response_text = response[0].get('generated_text', response[0].get('text', '')).strip()
                response_text = self._extract_relevant_answer(response_text)
            elif isinstance(response, str):
                response_text = self._extract_relevant_answer(response.strip())
            else:
                response_text = "Unexpected response format."
        except (KeyError, IndexError, TypeError) as e:
            print(f"Error processing response: {e}")
            response_text = "There was an error processing the response."
        
        return response_text  # Removed the "Response: Human:" prefix

    def _extract_relevant_answer(self, response_text):
        # Split the response text by newlines and filter out irrelevant parts
        lines = response_text.split('\n')
        relevant_lines = [line for line in lines if line.strip()]
        # Join the relevant lines into a single string
        return ' '.join(relevant_lines)  # Join all lines without truncating


    def _document_splitter(self, documents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = splitter.split_documents(documents)
        if not chunks:
            print("No chunks created. Check your document splitting settings.")
        return chunks

    def _get_embedding_func(self):
        embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        return embeddings

    def _initialize_vectorDB(self):
        self.vectordb = Chroma(
            persist_directory=self.db_path,
            embedding_function=self._get_embedding_func(),
        )

# Example usage:
# rag_system = RAGSystem()
# rag_system.process_documents(your_docs)
# response = rag_system.answer_query("Your query here")
# print(response)
