import os
import numpy as np
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()

class HybridRetriever:
    def __init__(self, db_path="./chroma_db"):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_db = Chroma(persist_directory=db_path, embedding_function=self.embeddings)

        # Initialize BM25
        self.all_texts = self.vector_db.get()['documents']
        self.tokenized_corpus = [doc.lower().split() for doc in self.all_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # Initialize Cross-Encoder Reranker
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def retrieve(self, query, k=3):
        # 1. Vector Search
        vector_results = self.vector_db.similarity_search(query, k=k*2)
        vector_docs = [doc.page_content for doc in vector_results]

        # 2. BM25 Search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:k*2]
        bm25_docs = [self.all_texts[i] for i in top_bm25_indices]

        # 3. Simple Reciprocal Rank Fusion (RRF)
        combined_scores = {}

        for rank, doc in enumerate(vector_docs):
            combined_scores[doc] = combined_scores.get(doc, 0) + 1 / (rank + 60)

        for rank, doc in enumerate(bm25_docs):
            combined_scores[doc] = combined_scores.get(doc, 0) + 1 / (rank + 60)

        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, score in sorted_docs[:k*2]]

        # 4. Reranking Step
        # Pair the query with each top document for the Cross-Encoder
        pairs = [[query, doc] for doc in top_docs]
        rerank_scores = self.reranker.predict(pairs)

        # Sort by reranker scores
        reranked_results = sorted(zip(top_docs, rerank_scores), key=lambda x: x[1], reverse=True)

        return [doc for doc, score in reranked_results[:k]]

if __name__ == "__main__":
    retriever = HybridRetriever()
    query = "What are the technical skills?"
    results = retriever.retrieve(query)
    for i, res in enumerate(results):
        print(f"Result {i+1}:\n{res}\n---")
