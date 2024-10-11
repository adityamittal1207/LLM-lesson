# Import necessary libraries
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser

api_key = ""

llm = ChatOpenAI(openai_api_key=api_key)

output_parser = StrOutputParser()

# 1. Loading: Load document from a URL
url = "https://aws.amazon.com/what-is/retrieval-augmented-generation/"
web_loader = WebBaseLoader(url)
documents = web_loader.load()  # Load the document from the URL

# 2. Splitting: Use Recursive Text Splitter for better chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Adjust sizes as needed
split_documents = text_splitter.split_documents(documents)  # Split documents into chunks

# 3. Embedding: Create embeddings for the documents using Hugging Face
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# 4. Vector Storing: Store the embeddings in a vector database
vector = Chroma.from_documents(split_documents, embeddings)  # Use split_documents for embedding

# 5. Retrieving: Setup the retriever
retriever = vector.as_retriever()

# 6. Generating a response from Phi-3 based off a prompt
prompt = ChatPromptTemplate.from_template("""Question: {input} Use the attached context to base your response: {context}""")

# Create the document chain
document_chain = create_stuff_documents_chain(llm, prompt)

# Create the retrieval chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Invoke the chain and parse the output
response = retrieval_chain.invoke({"input": "What is the advantage of using RAG?"})

print(response['answer'])
