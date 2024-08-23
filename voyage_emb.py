from dotenv import load_dotenv
load_dotenv()

import os
import voyageai
from langchain_voyageai import VoyageAIEmbeddings

api_key=os.environ.get('VOYAGE_API_KEY')
vo = voyageai.Client(api_key=api_key)


documents = [
    "Caching embeddings enables the storage or temporary caching of embeddings, eliminating the necessity to recompute them each time.",
    "An LLMChain is a chain that composes basic LLM functionality. It consists of a PromptTemplate and a language model (either an LLM or chat model). It formats the prompt template using the input key values provided (and also memory key values, if available), passes the formatted string to LLM and returns the LLM output.",
    "A Runnable represents a generic unit of work that can be invoked, batched, streamed, and/or transformed.",
]

documents_embeddings = vo.embed(
    documents, model="voyage-large-2-instruct", input_type="document"
).embeddings

embeddings = VoyageAIEmbeddings(
    voyage_api_key=api_key, model="voyage-large-2-instruct"
)

query = "What is LLM?"

# Get the embedding of the query
query_embedding = vo.embed([query], model="voyage-large-2-instruct", input_type="query").embeddings[0]


# Reranking
documents_reranked = vo.rerank(query, documents, model="rerank-lite-1", top_k=3)

for r in documents_reranked.results:
    print(f"Document: {r.document}")
    print(f"Relevance Score: {r.relevance_score}")
    print(f"Index: {r.index}")
    print()
# voyage_emb_langchain


from dotenv import load_dotenv
load_dotenv()

import os
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.retrievers import KNNRetriever

api_key=os.environ.get('VOYAGE_API_KEY')


documents = [
    "Caching embeddings enables the storage or temporary caching of embeddings, eliminating the necessity to recompute them each time.",
    "An LLMChain is a chain that composes basic LLM functionality. It consists of a PromptTemplate and a language model (either an LLM or chat model). It formats the prompt template using the input key values provided (and also memory key values, if available), passes the formatted string to LLM and returns the LLM output.",
    "A Runnable represents a generic unit of work that can be invoked, batched, streamed, and/or transformed.",
]

embeddings = VoyageAIEmbeddings(
    voyage_api_key=os.environ.get('VOYAGE_API_KEY'), model="voyage-large-2-instruct"
)

documents_embds = embeddings.embed_documents(documents)

query = input("Enter your query: ")

# Get the embedding of the query
query_embd = embeddings.embed_query(query)

retriever = KNNRetriever.from_texts(documents, embeddings)

# retrieve the most relevant documents
result = retriever.invoke(query)
top1_retrieved_doc = result[0].page_content  # return the top1 retrieved result

print(top1_retrieved_doc)
 