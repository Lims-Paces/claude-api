from dotenv import load_dotenv
load_dotenv()

import os
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.retrievers import KNNRetriever
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load API key from environment variables
api_key = os.environ.get('VOYAGE_API_KEY')

# Predefined documents
documents = [
    "Caching embeddings enables the storage or temporary caching of embeddings, eliminating the necessity to recompute them each time.",
    "An LLMChain is a chain that composes basic LLM functionality. It consists of a PromptTemplate and a language model (either an LLM or chat model). It formats the prompt template using the input key values provided (and also memory key values, if available), passes the formatted string to LLM and returns the LLM output.",
    "A Runnable represents a generic unit of work that can be invoked, batched, streamed, and/or transformed.",
]

# Initialize embeddings
embeddings = VoyageAIEmbeddings(
    voyage_api_key=api_key, model="voyage-large-2-instruct"
)

# Embed documents
documents_embds = embeddings.embed_documents(documents)

query = input("Enter your query: ")

# Get the embedding of the query
query_embd = embeddings.embed_query(query)

# Initialize retriever
retriever = KNNRetriever.from_texts(documents, embeddings)

# Retrieve the most relevant documents
result = retriever.invoke(query)


# Extract and print the top retrieved document
top1_retrieved_doc = result[0]  # Directly use the result as the document
print(top1_retrieved_doc)
