import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain_community.vectorstores import chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')


def load_doc(file_path):
    try:
        loader = PyPDFLoader(file_path)
        doc = loader.load()
        print("Document loaded successfully")
        return doc
    
    except FileNotFoundError:
        print("File not found")
        return None
    
    except Exception as e:
        print(f"An error occured: {e}")
        return None


def split_doc(doc):
    if doc is not None:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        doc = text_splitter.split_documents(doc)
        print("Document slitted successfully")
        return doc 
    
    else:
        print("No document to split")
        return None


def Embedd_doc(doc):
    if doc is not None:
        try:
            Embedder = OllamaEmbeddings()
            Embedd_doc = Embedder.embed_documents(doc)
            print("Document embedded successfully")
            return Embedd_doc
        
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        
    else:
        print("No document to embedd")
        return None
    

def EmbeddDoc_to_db(Embedd_doc):
    if Embedd_doc is not None:
        try:
            db = chroma()
            db_doc = db.from_documents(Embedd_doc)
            retriver = db_doc.as_retriever()
            print("Embedded doc saved to database")
            return retriver
        
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        
    else:
        print("No document to save")
        return None
    

def Create_chat_prompt_template(query):
    if query:
        prompt = ChatPromptTemplate.from_template("""
        Answer the following question and i will give you a reward
        <context>
        {context}
        </context>
        Question: {query}""")
        return prompt
    
    else:
        print("Query is not provided")
        return None


def load_model(model_name):
    try:
        model_ = Ollama(model=model_name)
        print("Model loaded successfully")
        return model_
    
    except Exception as e:
        print(f"An error occurred while loading model: {e}")
        return None
    

def stuff_doc_chain(prompt, model_):
    if prompt and model_ is not None:
        document_cahin = create_stuff_documents_chain(model_, prompt)
        print("Stuff doc chain created successfully")
        return document_cahin
    
    else:
        print("Error in creating doc chain")
        return None


def retrival_chain(retriver, document_chain):
    try:
        retrival_chain = create_retrieval_chain(retriver, document_chain)
        print("Retrival chain created successfully")
        return retrival_chain
    
    except Exception as e:
        print(f"Error in creating retrival chain: {e}")
        return None
    



doc = load_doc('C:\\Users\\FINE\\Desktop\\Mlops\\New DOCX Document.pdf')
split_docs = split_doc(doc)
embedded_docs = Embedd_doc(split_docs)
retriever = EmbeddDoc_to_db(embedded_docs)

query = "What is transformers"
prompt = Create_chat_prompt_template(query)

model_name = "llama3"
model = load_model(model_name)

document_chain = stuff_doc_chain(prompt, model)
retrieval_chain = retrival_chain(retriever, document_chain)

if retrieval_chain:
    response = retrieval_chain.invoke({'query':query})
    print('Response: ', response)