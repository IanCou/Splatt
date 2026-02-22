import os
import logging
from typing import List, Optional
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ConstructionSafetyRAG:
    """
    RAG system to evaluate construction site safety using LangChain and Google Gemini.
    """

    def __init__(self, pdf_path: str = "construction_safety.pdf", index_path: str = "faiss_index", model_name: str = "gemini-2.5-flash"):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        self.pdf_path = pdf_path
        self.index_path = index_path
        self.model_name = model_name
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=self.api_key)
        self.llm = ChatGoogleGenerativeAI(model=self.model_name, google_api_key=self.api_key, temperature=0)

        self.vector_store = None
        self.retriever = None
        self.rag_chain = None

    def initialize(self):
        """
        Loads the PDF, chunks the text, and initializes the vector store and RAG chain.
        Tries to load an existing index first.
        """
        if os.path.exists(self.index_path):
            logger.info(f"Loading existing FAISS index from {self.index_path}...")
            self.vector_store = FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            if not os.path.exists(self.pdf_path):
                logger.error(f"PDF file not found: {self.pdf_path}")
                raise FileNotFoundError(f"Construction safety knowledge base not found at {self.pdf_path}")

            logger.info(f"Loading PDF from {self.pdf_path}...")
            loader = PyPDFLoader(self.pdf_path)
            docs = loader.load()
            
            # Limit to first 20 pages for demo/quota purposes
            if len(docs) > 20:
                logger.info(f"Limiting to first 20 pages from {len(docs)} total pages for demo.")
                docs = docs[:20]

            logger.info("Splitting documents into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            logger.info("Initializing and saving FAISS vector store...")
            self.vector_store = FAISS.from_documents(documents=splits, embedding=self.embeddings)
            self.vector_store.save_local(self.index_path)

        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})

        # Setup RAG chain
        prompt = ChatPromptTemplate.from_template("""
        You are a construction site safety expert. Use the following context from safety manuals to determine if the described scene is dangerous.
        
        Context:
        {context}
        
        Scene Description:
        {description}
        
        Task:
        Strictly respond with ONLY one word: "Safe" if the scene follows safety protocols, or "Unsafe" if there are any potential hazards or violations. Do not provide any other text.
        """)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.rag_chain = (
            {"context": self.retriever | format_docs, "description": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        logger.info("RAG chain initialized successfully.")

    def evaluate_safety(self, scene_description: str) -> str:
        """
        Evaluates the safety of a scene description using RAG.
        """
        if not self.rag_chain:
            logger.info("Initializing RAG system...")
            self.initialize()

        logger.info(f"Evaluating safety for: {scene_description[:100]}...")
        return self.rag_chain.invoke(scene_description)

if __name__ == "__main__":
    # Test script
    try:
        safety_system = ConstructionSafetyRAG()
        description = "A worker is standing on a high ledge without a harness while holding a heavy tool."
        result = safety_system.evaluate_safety(description)
        print("\n--- Safety Analysis ---")
        print(result)
    except Exception as e:
        print(f"Error: {e}")
