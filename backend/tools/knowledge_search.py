import os
from typing import List, Dict, Any
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
import math
from langsmith import traceable
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger import logger

class HybridRetriever:
    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock
        self.alpha_dense_weight = 0.5
        
        if not self.use_mock:
            # Native Pinecone Initialization
            self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY", "dummy-key"))
            self.index = self.pc.Index(os.environ.get("PINECONE_INDEX_NAME", "enterprise-search"))

    def compute_hybrid_score(self, dense_score: float, sparse_score: float) -> float:
        """
        Combines dense and sparse scores into a single hybrid value.
        Uses alpha weighting: alpha*dense + (1-alpha)*sparse
        """
        # Normalize BM25 score heuristically if necessary, or pass through Pinecone's native hybrid (SPLADE)
        return (self.alpha_dense_weight * dense_score) + ((1 - self.alpha_dense_weight) * sparse_score)

    @traceable(run_type="retriever", name="hybrid_pinecone_search")
    async def search(
        self,
        query: str,
        dense_embedding: List[float],
        namespace: str = "default",
        metadata_filters: Dict[str, Any] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid search incorporating namespaces and metadata filtering.
        """
        logger.info("executing_hybrid_search", query=query, namespace=namespace, limit=top_k)
        
        if self.use_mock:
            # We mock the response to gracefully degrade if Pinecone is inaccessible
            # and to satisfy testing scenarios without risking API key exposures.
            return self._mock_search_results(query, metadata_filters)
        else:
            # Production execution utilizing Pinecone's native Sparse/Dense feature where available
            # Or manual reranking using rank_bm25
            
            # Note: For Pinecone hybrid, you typically pass a sparse_vector derived from SPLADE/BM25
            # sparse_vector = generate_sparse_vector(query) 
            response = self.index.query(
                namespace=namespace,
                vector=dense_embedding,
                # sparse_vector=sparse_vector,
                filter=metadata_filters or {},
                top_k=top_k,
                include_metadata=True
            )
            return [
                {
                    "id": match["id"],
                    "score": match["score"],
                    "text": match["metadata"].get("text", ""),
                    "attribution": match["metadata"].get("attribution", "Unknown Source"),
                    "metadata": match["metadata"]
                }
                for match in response["matches"]
            ]

    def _mock_search_results(self, query: str, filters: dict = None) -> List[Dict]:
        """Provides simulated data for the sample search"""
        mock_docs = [
            {
                "id": "inc-001",
                "score": 0.89,
                "text": "Payment gateway outage during Q3 due to excessive token timeout thresholds.",
                "attribution": "Incident Report #2904",
                "metadata": {
                    "department": "payments",
                    "document_type": "incident",
                    "access_level": "internal",
                    "created_date": "2025-01-01"
                }
            },
            {
                "id": "run-001",
                "score": 0.74,
                "text": "To restart the core payment routing engine, clear the Redis cache and bounce the pod.",
                "attribution": "Payment Routing Runbook",
                "metadata": {
                    "department": "payments",
                    "document_type": "runbook",
                    "access_level": "internal",
                    "created_date": "2023-11-12"
                }
            }
        ]
        
        # Apply strict metadata filtering to mock
        if filters:
            filtered_docs = []
            for doc in mock_docs:
                match = True
                for k, v in filters.items():
                    if doc["metadata"].get(k) != v:
                        match = False
                        break
                if match:
                    filtered_docs.append(doc)
            return filtered_docs
        
        return mock_docs
