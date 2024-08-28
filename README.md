# Rag_pipeline

Document Processing and Retrieval System
This script provides a pipeline for processing PDF documents and retrieving information using LangChain and Ollama embeddings.

Prerequisites
Python 3.7 or later
Necessary Python packages: dotenv, langchain_community, langchain_text_splitters, langchain_core, chroma (or an equivalent vector store library), and Ollama (or an equivalent embeddings library).
Setup
Install the required Python packages using pip.

Create a .env file in the same directory as the script with your LangChain API key:

LANGCHAIN_API_KEY=your_langchain_api_key

Overview
Functions
Load Document: Loads a PDF document from a specified path.
Split Document: Divides the document into chunks for easier processing.
Embed Document: Converts text chunks into embeddings.
Save to Database: Stores embeddings in a vector database and provides a retriever.
Create Prompt Template: Generates a template for querying the model.
Load Model: Loads a specified language model.
Create Document Chain: Sets up a document chain using the prompt and model.
Create Retrieval Chain: Constructs a retrieval chain to fetch relevant information.

Usage
Load a PDF document from the filesystem.
Split the document into manageable chunks.
Embed the document chunks to convert them into embeddings.
Store the embeddings in a vector database and obtain a retriever.
Create a chat prompt template for querying.
Load the language model for processing the prompt and embeddings.
Create a document chain using the prompt and model.
Construct a retrieval chain for querying and retrieving relevant information.
Invoke the retrieval chain with a query to get the response.

Notes
Ensure the PDF file path is correct and accessible.
Adjust model names and parameters according to your needs.
Verify the LangChain API key in the .env file and ensure all dependencies are installed.

Troubleshooting
Check the file path and permissions if loading the document fails.
Ensure the LangChain API key is correctly set.
Make sure all required packages are installed and up to date.
