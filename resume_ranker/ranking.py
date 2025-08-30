from __future__ import annotations

from typing import List, Dict, Any

import numpy as np
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.docstore.document import Document
import chromadb
import json

# In production you may want to inject these dependencies via DI for testing.

class ResumeRanker:
    """Advanced ranking combining embedding similarity search with LLM scoring."""

    def __init__(
        self,
        openai_api_key: str,
        model: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-small",
        collection_name: str = "resumes",
    ) -> None:
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model=embedding_model)
        self.llm = ChatOpenAI(model=model, temperature=0.3, openai_api_key=openai_api_key)
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(collection_name)

    def add_resumes(self, resume_texts: List[str], metadatas: List[Dict[str, Any]]):
        """Embed and store resumes in vector DB."""
        vectors = self.embeddings.embed_documents(resume_texts)
        ids = [str(i) for i in range(len(vectors))]
        self.collection.add(ids=ids, embeddings=vectors, metadatas=metadatas)

    def rank(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Return ranked resumes combining similarity and LLM detailed analysis."""
        query_embedding = self.embeddings.embed_query(query)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k, include=["embeddings", "metadatas"])

        # Flatten
        retrieved = []
        for idx, metadata in enumerate(results["metadatas"][0]):
            score = results["distances"][0][idx]
            retrieved.append({"metadata": metadata, "similarity": 1 - score})

        # Compose prompt to LLM for each resume and get refined score
        enriched: List[Dict[str, Any]] = []
        for item in retrieved:
            resume_text = item["metadata"]["resume_text"][:50000]
            prompt = (
                "You are a senior recruiter. The following is a job/criteria description: \n" + query +
                "\n---\nHere is the candidate resume:\n" + resume_text +
                "\nGive a score 0-100 and a brief explanation JSON: {\"match_score\": <int>, \"reasoning\": <string>}"
            )
            response = self.llm.invoke(prompt).content
            try:
                parsed = json.loads(response)
            except Exception:
                parsed = {"match_score": 0, "reasoning": response[:200]}
            enriched.append({**item["metadata"], **parsed, "similarity": item["similarity"]})

        # Final sort by combined score (70% LLM score, 30% similarity)
        def combined(e):
            return 0.7 * e.get("match_score", 0) + 0.3 * (e.get("similarity", 0) * 100)

        return sorted(enriched, key=combined, reverse=True)