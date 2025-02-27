## **Retrieval-Augmented Generation (RAG) pipeline**

### **Step 1: Build the Vector Store**
- The documents are first processed by an **Embedding Model** that converts text into numerical vector representations (embeddings).
- These embeddings are then stored in a **Vector Database (Vector Store)**, enabling efficient similarity searches.

### **Step 2: Cache, Search, and Re-rank**
1. A **user query** is first checked against an **Index Search Cache**.
   - If the query has been searched before, the cache returns the stored results quickly.
   - If the query is not in the cache, it is passed to the main **Vector Database** for searching.
2. The **Vector Database** performs a similarity search to retrieve the **top-k closest document chunks** related to the query.
3. The retrieved chunks are then **re-ranked using a Cross-Encoder model** to improve the relevance of the results.
4. The **top 3 ranked documents** are then selected for final processing.

### **Step 3: Generative Search (Final Answer Generation)**
- The **query, prompt, and top 3 retrieved documents** are passed to an **LLM (Large Language Model)** (e.g., GPT-3.5 or GPT-4).
- The **LLM generates a well-structured response** based on the retrieved documents and query.
- The response is finally **returned to the user**, along with citations if necessary.

---
## **Implementation Guide**
### **1. Build the Vector Store**
**Install required libraries:**
```bash
pip install sentence-transformers chromadb openai
```

**Convert documents into embeddings and store them in a vector database:**
```python
from sentence_transformers import SentenceTransformer
import chromadb

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB for storing embeddings
client = chromadb.PersistentClient(path="./vector_db")
collection = client.get_or_create_collection(name="insurance-docs")

# Example document chunks
documents = ["Life insurance covers total disability.", "Accidental death benefits vary."]
embeddings = embedder.encode(documents)

# Store in vector database
for i, doc in enumerate(documents):
    collection.add(ids=[str(i)], documents=[doc], embeddings=[embeddings[i].tolist()])
```

---
### **2. Cache, Search, and Re-rank**
**Define search function:**
```python
def search(query):
    query_embedding = embedder.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=10)
    return results
```

**Apply Cross-Encoder re-ranking:**
```python
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def apply_cross_encoder(query, results):
    cross_inputs = [[query, doc] for doc in results["documents"][0]]
    scores = cross_encoder.predict(cross_inputs)
    results["scores"] = scores
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return {key: [results[key][0][i] for i in sorted_indices] for key in results}
```

---
### **3. Generate Final Answer with LLM**
**Pass the query and retrieved documents to GPT-3.5 for final response generation:**
```python
import openai

def generate_response(query, results):
    top_docs = "
".join(results["documents"][:3])
    prompt = f"""You are an expert in insurance policies.
    Answer the following query based on the given documents:
    
    Query: {query}
    
    Documents:
    {top_docs}
    
    Provide a concise answer with citations.
    """
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ])
    return response["choices"][0]["message"]["content"]
```

---
### **Final Execution**
```python
query = "What is the life insurance coverage for disability?"
retrieved_results = search(query)
reranked_results = apply_cross_encoder(query, retrieved_results)
response = generate_response(query, reranked_results)
print(response)
```

This pipeline ensures efficient retrieval, ranking, and answer generation, providing highly relevant and well-cited responses using **Retrieval-Augmented Generation (RAG).**

Would you like additional enhancements, such as fine-tuning for better results or integrating a UI?

## Explanation

 ([image]()) *Overview of a Retrieval-Augmented Generation (RAG) pipeline architecture.* The RAG pipeline combines information retrieval with text generation to produce accurate, context-aware answers. It involves first retrieving relevant context (documents or data) from a knowledge base (often a *vector store*) and then using a generative model (LLM) to formulate the answer using that context ([Building advanced RAG Pipelines for high-accuracy responses](https://www.confidentialmind.com/post/building-advanced-rag-pipelines#:~:text=,readable%20response)). This approach grounds the large language model with external knowledge, reducing hallucinations and improving factual accuracy. The pipeline can be divided into three main stages: **Vector Store**, **Search & Re-rank**, and **Generative Search** (answer generation), with additional components (embedding model, cross-encoder re-ranker, cache, etc.) playing specific roles in each stage.

### Vector Store (Building the Knowledge Base)  
This stage prepares and stores the knowledge that the system will draw upon. First, an **embedding model** (typically a bi-encoder like a sentence transformer) converts each document or text chunk in your dataset into a numerical vector representation. These embeddings capture the semantic meaning of the text, so that semantically similar texts have vectors that are close together in vector space ([Using Cross-Encoders as reranker in multistage vector search | Weaviate](https://weaviate.io/blog/cross-encoders-as-reranker#:~:text=Vector%20databases%20like%20Weaviate%20use,closest%20to%20the%20query%20embedding)). All document vectors are stored in a **vector database** (e.g. ChromaDB, Pinecone, FAISS) which indexes them for fast similarity search. The vector store acts as the knowledge base: it allows the system to find which documents are most relevant to a new query by comparing embedding vectors. In practice, you would preprocess raw documents by splitting them into smaller chunks (to improve retrieval granularity) and then embed and load them into the vector DB along with metadata (like document IDs or sources). The result is a semantic index of your data that can be queried efficiently. *Role of components:* The **embedding model** creates the vectors (for both documents and queries), and the **vector database** holds those vectors and supports similarity search. This step is usually done offline or ahead of time for all documents, so that at query time you have a prepared vector store to search. 

### Search & Re-rank (Retrieval with Cross-Encoder Refinement)  
When a user asks a question (the query), the system performs a retrieval in two phases. **First**, the query is embedded into a vector using the same embedding model as above ([Building advanced RAG Pipelines for high-accuracy responses](https://www.confidentialmind.com/post/building-advanced-rag-pipelines#:~:text=,readable%20response)). The vector database is then queried for the nearest neighbor embeddings – i.e. it returns the top *K* document chunks whose vectors are most similar to the query vector. This initial similarity search (using the embeddings) is very fast and serves as a coarse filter to get relevant candidates. However, the top results from a pure vector similarity search might not be perfectly ordered by relevance, so we apply a re-ranking step for higher accuracy ([A Hands-on Guide to Enhance RAG with Re-Ranking](https://adasci.org/a-hands-on-guide-to-enhance-rag-with-re-ranking/#:~:text=The%20generator%20produces%20a%20final,rich%20and%20highly%20relevant%20documents)). In the **second phase**, a **cross-encoder** model re-ranks those *K* results. A cross-encoder (often a BERT or transformer-based model fine-tuned for pairwise relevance) takes the **query and a candidate document together** as input and assigns a relevance score to the pair ([Using Cross-Encoders as reranker in multistage vector search | Weaviate](https://weaviate.io/blog/cross-encoders-as-reranker#:~:text=Bi,the%20query%20and%20data%20object)) ([Using Cross-Encoders as reranker in multistage vector search | Weaviate](https://weaviate.io/blog/cross-encoders-as-reranker#:~:text=If%20a%20Cross,to%20perform%20the%20search)). Unlike the bi-encoder embedding model which assessed similarity in vector space, the cross-encoder actually reads the full text of the query and document and makes a judgment, which is more accurate but also more computationally expensive ([Using Cross-Encoders as reranker in multistage vector search | Weaviate](https://weaviate.io/blog/cross-encoders-as-reranker#:~:text=If%20a%20Cross,to%20perform%20the%20search)). Therefore, we only use it on a small number of candidates. The cross-encoder scores all retrieved candidates and we then sort them by score, choosing the top few (e.g. top 1 or top 3) as the most relevant context. This multi-stage retrieval gives us the **speed** of embedding-based search and the **accuracy** of a deeper relevance model ([Using Cross-Encoders as reranker in multistage vector search | Weaviate](https://weaviate.io/blog/cross-encoders-as-reranker#:~:text=Catching%20fish%20with%20the%20big,this%20on%20large%20scale%20datasets)). The re-ranking step is crucial for quality: it significantly improves the relevance of the documents passed to the LLM, which in turn helps the final answer be accurate and specific ([A Hands-on Guide to Enhance RAG with Re-Ranking](https://adasci.org/a-hands-on-guide-to-enhance-rag-with-re-ranking/#:~:text=In%20conclusion%2C%20re,selecting%20the%20optimal%20reranking%20model)). *Role of components:* The **embedding model** is used here again to encode the user’s query for the initial search. The **cross-encoder** is a separate model specialized for relevance ranking – it does not produce embeddings for indexing, but directly evaluates query-document pairs to refine the results. Together, the retrieval (vector search + re-rank) stage finds and prioritizes the information that the generative model will use.

### Generative Search (Answer Generation with LLM)  
In the final stage, the top ranked document snippets from the retrieval phase are passed along with the user’s query to a **large language model (LLM)**, such as GPT-3.5. The LLM (the generative component) is prompted with the relevant context and asked to produce a final answer. Essentially, the model is doing a *“guided”* generation: it uses the retrieved documents as additional context to formulate its response ([Building advanced RAG Pipelines for high-accuracy responses](https://www.confidentialmind.com/post/building-advanced-rag-pipelines#:~:text=,readable%20response)). A prompt for the LLM might include the retrieved text (or a summarized version of it) plus the question, instructing the model to base its answer on the given information. Because the LLM now has access to factual context, it can produce a more accurate and grounded answer than it would on its own. The term “Generative Search” refers to this process of using a generative model to answer the query, as opposed to returning a document list. The LLM will combine information from the provided documents and generate a coherent answer in natural language ([A Hands-on Guide to Enhance RAG with Re-Ranking](https://adasci.org/a-hands-on-guide-to-enhance-rag-with-re-ranking/#:~:text=The%20generator%20produces%20a%20final,rich%20and%20highly%20relevant%20documents)). *Role of components:* The **LLM** is the core of this stage – it’s responsible for understanding the question in context of the retrieved info and composing a helpful answer. It’s still a large neural network model (GPT-style) but now augmented with knowledge from the vector store. Often, you might use a prompt template that ensures the model stays factual (e.g. *“Use the following context to answer the question…”* and instruct it not to fabricate information). The **cache** (if implemented) can also play a role here: for example, you might cache the final answer for a given query to serve identical questions faster in the future.

### Cache (Optimization Layer)  
A **cache** is not a mandatory part of the RAG pipeline, but it is a common addition to improve efficiency, especially in production deployments. There are two main caching strategies in RAG: **embedding caching** and **query/response caching**. *Embedding caching* means storing the embeddings that have been computed before (for documents or for frequent queries) so you don’t recompute them on every run ([Advanced RAG Implementation on Custom Data Using Hybrid Search, Embed Caching And Mistral-AI | by Plaban Nayak | AI Planet](https://medium.aiplanet.com/advanced-rag-implementation-on-custom-data-using-hybrid-search-embed-caching-and-mistral-ai-ce78fdae4ef6#:~:text=Embeddings%20can%20be%20stored%20or,avoid%20needing%20to%20recompute%20them)). For example, once you embed all your documents and load them into the vector store, those embeddings can be persisted (ChromaDB can be configured to persist to disk, or you can use an embedding cache in memory or a key-value store). Similarly, if your system sees the same query multiple times, you can cache the query’s embedding vector or even the retrieved results. This saves the cost of re-embedding the text and doing the full search again. *Query/response caching* goes a step further: if the exact same question was already asked and you have a cached answer (along with the supporting context used), you can directly return the cached answer (possibly after a quick validity check) instead of running the pipeline anew. By implementing caching, one can **significantly reduce latency and costs** in RAG systems, avoiding repetitive work ([Advanced RAG Implementation on Custom Data Using Hybrid Search, Embed Caching And Mistral-AI | by Plaban Nayak | AI Planet](https://medium.aiplanet.com/advanced-rag-implementation-on-custom-data-using-hybrid-search-embed-caching-and-mistral-ai-ce78fdae4ef6#:~:text=Embeddings%20can%20be%20stored%20or,avoid%20needing%20to%20recompute%20them)). The cache must be carefully managed – for instance, cache entries should be invalidated if the underlying knowledge data changes (to avoid serving outdated information). In summary, the **cache** serves as a speed-up mechanism: the pipeline will first check the cache (for an embedding or an answer) and use it if available, otherwise it will proceed with the normal steps and then store the new result in the cache for next time.

## Implementation Guide

Below is a step-by-step guide to implement a RAG pipeline with Python. We’ll use **ChromaDB** as the vector store, a sentence-transformer model for embeddings, a cross-encoder for re-ranking, and OpenAI’s GPT-3.5 for final answer generation. Each step includes code snippets and explanations:

### 1. Setting up the Embedding Model and Vector Database  
First, install or import the required libraries. We need a sentence transformer model for embeddings and the ChromaDB client for our vector store. For example, we can use the `"all-MiniLM-L6-v2"` model from SentenceTransformers (a lightweight, fast embedding model), and create a ChromaDB collection to store document embeddings.

```python
# Import the embedding model and the vector DB client
from sentence_transformers import SentenceTransformer
import chromadb

# Load the embedding model (bi-encoder) for text
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize ChromaDB client and create a new collection for documents
client = chromadb.Client()
collection = client.create_collection(name="knowledge_base")
```

Next, prepare your documents. In a real scenario, you might load documents from files or a database, then split them into chunks of a few hundred words (to fit into vector and LLM context size). For this example, let's assume we have a small list of text documents ready:

```python
# Example documents (in practice, load and preprocess your data)
docs = [
    "London is the capital of England and is known for the River Thames.",
    "Paris is the capital city of France, famous for the Eiffel Tower.",
    "The Thames River flows through southern England, including London.",
]
doc_ids = ["doc1", "doc2", "doc3"]

# Compute embeddings for each document using the embedding model
doc_embeddings = embedding_model.encode(docs)  # returns a list/array of vectors

# Add documents and their embeddings to the Chroma vector store
collection.add(documents=docs, embeddings=doc_embeddings, ids=doc_ids)
print(f"Added {collection.count()} documents to the vector store.")
```

Here, each document text is converted to a vector (`doc_embeddings`), and then stored in the Chroma collection along with an ID. The `collection.count()` should confirm the number of items indexed. Now we have a vector store that can be queried – essentially a knowledge base of embeddings ([A Hands-on Guide to Enhance RAG with Re-Ranking](https://adasci.org/a-hands-on-guide-to-enhance-rag-with-re-ranking/#:~:text=Let%20us%20use%20OpenAIEmbeddings%20to,it%20in%20a%20vector%20database)).

### 2. Searching for Relevant Documents  
With the vector store ready, we can handle incoming queries. For a given user query, we embed the query text into a vector using the same embedding model. Then we perform a similarity search in ChromaDB to retrieve the top relevant document vectors. Chroma will return the documents (and optionally their similarity scores or metadata).

```python
# User query (question we want to answer)
query = "What is the capital of England?"

# Embed the query into a vector
query_embedding = embedding_model.encode([query])[0]  # encode returns a list; take the first element

# Perform vector search in the collection for the top-5 similar documents
results = collection.query(query_embeddings=[query_embedding], n_results=5, include=["documents", "distances"])

# Extract the retrieved documents and distances
retrieved_docs = results["documents"][0]   # list of document texts for this query
retrieved_scores = results["distances"][0] # list of similarity distances (lower means more similar)
for doc, distance in zip(retrieved_docs, retrieved_scores):
    print(f"Retrieved doc: '{doc[:50]}...', distance: {distance:.4f}")
```

In this snippet, `collection.query` searches the vector store using the query embedding. We ask for `n_results=5` documents, but in our small example we only have 3 docs, so it will return those. The `include=["documents", "distances"]` instructs Chroma to return the actual document texts and the distance scores. We print out the retrieved documents (truncated for display) and their distances to see how similar they are. At this stage, the results are ranked by vector similarity. The document with the smallest distance (or highest cosine similarity) is considered the most relevant according to the embedding model. 

### 3. Re-ranking the Results with a Cross-Encoder  
Now we will refine the ordering of these retrieved documents using a cross-encoder re-ranker. We load a cross-encoder model (for example, the `"cross-encoder/ms-marco-MiniLM-L-6-v2"` model, which is trained on MS MARCO for question-answering relevance). The cross-encoder will take each candidate document along with the query and output a relevance score. We then use those scores to sort the documents in order of relevance.

```python
from sentence_transformers import CrossEncoder

# Load a cross-encoder model for re-ranking (it will output a relevance score for a (query, doc) pair)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Prepare (query, document) pairs for each retrieved document
pairs = [(query, doc) for doc in retrieved_docs]

# Get relevance scores for each pair
scores = reranker.predict(pairs)

# Combine documents with their scores and sort by score descending
scored_docs = list(zip(scores, retrieved_docs))
scored_docs.sort(key=lambda x: x[0], reverse=True)

# The documents are now re-ranked by the cross-encoder scores
reranked_docs = [doc for score, doc in scored_docs]
print("Top document after re-ranking:", reranked_docs[0][:50], "...")
```

After running the cross-encoder, we obtain a list of `scores` (e.g., higher score means more relevant). We then sort the documents by these scores. The variable `reranked_docs` is an ordered list of document texts from most relevant to least, according to the cross-encoder. In many cases, you might only keep the very top result or top few results for the final answer generation, since the LLM has a context size limit. This two-step retrieval (vector search + cross-encoder) ensures we got the most relevant context for the query ([Using Cross-Encoders as reranker in multistage vector search | Weaviate](https://weaviate.io/blog/cross-encoders-as-reranker#:~:text=Catching%20fish%20with%20the%20big,this%20on%20large%20scale%20datasets)). We have leveraged the speed of the embedding-based search to narrow down candidates, and then the accuracy of the cross-encoder to pick the best of those candidates. This should improve the quality of the answer, as the model will see the most pertinent information ([A Hands-on Guide to Enhance RAG with Re-Ranking](https://adasci.org/a-hands-on-guide-to-enhance-rag-with-re-ranking/#:~:text=In%20conclusion%2C%20re,selecting%20the%20optimal%20reranking%20model)).

*(Optional:* If you skip the re-ranking step, the pipeline will still work – you would just take the top result(s) from the raw vector search. However, using a re-ranker often boosts answer accuracy, at the cost of some extra computation. If latency is a concern or if the embedding model is already very accurate for your domain, you might choose to omit this step. Alternatively, you could use a lighter-weight cross-encoder or limit re-ranking to fewer documents to save time.)*

### 4. Generating the Final Answer with GPT-3.5  
With the relevant context in hand, we can now query a large language model to generate the answer. We will use OpenAI’s GPT-3.5 (via the `openai` package) for the generative step. We construct a prompt that includes the retrieved information and the user’s question, then ask the LLM to answer based on that. It’s important to clearly instruct the model to use the provided context.

```python
import openai
openai.api_key = "YOUR_OPENAI_API_KEY"  # replace with your actual API key

# Compose a prompt with the top retrieved document(s) as context
top_context = "\n".join(reranked_docs[:2])  # take the top 2 docs for context (if available)
prompt = (
    f"You are a knowledgeable assistant. Use the following context to answer the question.\n\n"
    f"Context:\n{top_context}\n\n"
    f"Question: {query}\n"
    f"Answer:"
)

# Call the OpenAI GPT-3.5 model (gpt-3.5-turbo) with the constructed prompt
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[ {"role": "user", "content": prompt} ]
)

# Extract the assistant's answer from the response
answer = response["choices"][0]["message"]["content"]
print("Answer:", answer)
```

In this code, we join the top two retrieved chunks into one `top_context` string (you could choose only the top 1, or more than 2 if the question is complex and the model can handle the length). The prompt clearly provides this context and the question. We then call the `ChatCompletion.create` with `gpt-3.5-turbo`, treating the whole prompt as the user message. The model’s answer is returned in the `response`, and we print it out. For example, if the query was "What is the capital of England?", and our documents contained that information, GPT-3.5 should output an answer like *"The capital of England is London."* using the context we supplied.

**Best practices for prompt design:** When crafting the prompt, it’s a good idea to remind the model to use **only** the given context for answering. This can reduce hallucinations. You might add a line like *"If the answer is not in the context, say you don't know."* to the instructions. Also, ensure the context is not too large for the model’s token limit – if you have many relevant documents, you may need to truncate or summarize some content. The OpenAI GPT-3.5 model has a few thousand token capacity (e.g., ~4096 tokens), which must include the prompt and the answer, so keep that in mind.

### 5. Optimizations for Efficiency and Accuracy  
To make the RAG pipeline efficient and robust, consider the following optimizations:

- **Embedding Cache:** Computing embeddings for every query or repeatedly for the same documents can be wasteful. Use a cache to store embeddings for texts that have been seen before ([Advanced RAG Implementation on Custom Data Using Hybrid Search, Embed Caching And Mistral-AI | by Plaban Nayak | AI Planet](https://medium.aiplanet.com/advanced-rag-implementation-on-custom-data-using-hybrid-search-embed-caching-and-mistral-ai-ce78fdae4ef6#:~:text=Embeddings%20can%20be%20stored%20or,avoid%20needing%20to%20recompute%20them)). For instance, you can hash the text of a query or document and use it as a key in a dictionary or a database. Before encoding a new text, check the cache – if the embedding exists, reuse it instead of recomputing. Libraries like LangChain provide utilities (e.g. `CacheBackedEmbeddings`) to automate this caching ([Advanced RAG Implementation on Custom Data Using Hybrid Search, Embed Caching And Mistral-AI | by Plaban Nayak | AI Planet](https://medium.aiplanet.com/advanced-rag-implementation-on-custom-data-using-hybrid-search-embed-caching-and-mistral-ai-ce78fdae4ef6#:~:text=Embeddings%20can%20be%20stored%20or,avoid%20needing%20to%20recompute%20them)), or you can implement it manually as a simple Python dict in memory for smaller applications.

- **Result Cache:** Similarly, you can cache the results of entire queries. If your app often receives identical questions, store the top retrieved docs and even the final answer for each unique query. Next time the same query comes, you can instantly return the cached answer (or at least skip straight to the generation step). This greatly reduces latency and OpenAI API calls for repeat queries. Make sure to update or invalidate this cache when your document data changes, so the answers stay up-to-date.

- **Chunking and Document Preparation:** Splitting documents into coherent chunks is crucial. If documents are too large, you might miss relevant info because only part of a document is similar to the query. Use a text splitter (by paragraph, sentence, or token count) to create chunks (e.g. 200-500 tokens each) when building the vector store. This improves the chance that a relevant piece of information is retrieved and fits in the LLM context window.

- **Choosing the Right Models:** There is a trade-off between speed and accuracy in each component. Smaller embedding models (like MiniLM used above) are fast but a larger model (like multi-GB ones or OpenAI’s embeddings) might capture nuances better, at the cost of speed ([Using Cross-Encoders as reranker in multistage vector search | Weaviate](https://weaviate.io/blog/cross-encoders-as-reranker#:~:text=In%20search%2C%20or%20semantic%20matching,to%20benefit%20from%20both%20models)). Similarly, the cross-encoder we used is fairly small; you could use a more powerful cross-encoder or even a larger reranker model for better accuracy, but that will slow down responses. Evaluate your needs: for high-throughput systems, you might favor smaller models or reduce the number of reranked documents. For high-stakes domains where accuracy is paramount, larger models and re-ranking a bigger candidate set might be justified. 

- **Hybrid Search:** For some applications, combining vector search with keyword search (BM25) can yield better results (this is called hybrid search). For example, you might retrieve some results by lexical search and some by vector similarity and merge them (many vector DBs or frameworks support this). This can catch exact matches (like specific codes or names) that pure semantic search might miss ([Building advanced RAG Pipelines for high-accuracy responses](https://www.confidentialmind.com/post/building-advanced-rag-pipelines#:~:text=,lack%20of%20embedding%20generation%20ability)). The retrieved results can all be fed into the reranker or LLM for consideration.

- **Monitoring and Iteration:** After deploying, monitor the quality of answers. If you find the LLM is still hallucinating or missing information, consider augmenting the prompt instructions, increasing the number of documents retrieved, or improving your document set (e.g., adding more sources or refining chunk sizes). You can also implement feedback loops where if the LLM expresses uncertainty, you adjust the query or retrieval process (e.g., query reformulation).

By following this guide, you set up a working RAG pipeline: your system vectorizes and stores knowledge, finds relevant pieces for any new question, optionally re-ranks them for relevance, and uses a powerful LLM to generate a well-informed answer. This architecture allows you to **deploy LLMs on custom data** with improved accuracy and control, since the answers are grounded in your provided documents rather than just the model’s own training data.