from pathlib import Path
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions

class RAGRetriever:
    def __init__(self, data_dir: str, profession: str, embedding_model: str = "all-MiniLM-L6-v2"):
        self.data_dir = Path(data_dir)
        self.profession = profession
        self.embedding_model_name = embedding_model
        
        self.db_dir = self.data_dir / "rag" / "vector_database" / profession
        
        if not self.db_dir.exists():
            raise ValueError(f"Vector database not found at {self.db_dir}. Please run ingestion first.")
            
        self.client = chromadb.PersistentClient(path=str(self.db_dir))
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        
        self.collection_name = f"rag_{self._sanitize_name(profession)}"
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_func
            )
        except Exception as e:
            raise ValueError(f"Collection {self.collection_name} not found in DB. Error: {e}")

    def _sanitize_name(self, name: str) -> str:
        return "".join(c if c.isalnum() else "_" for c in name)

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve top k relevant chunks for the query.
        Returns a list of dicts with 'text', 'source', 'score'.
        """
        # E5 models require "query: " prefix
        if "e5" in self.embedding_model_name.lower():
            query = f"query: {query}"
        


        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        
        # Unpack results
        # results is a dict of lists (one list per query)
        # We only have one query
        
        retrieved_chunks = []
        if results['documents']:
            docs = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0] if results['distances'] else [0.0] * len(docs)
            
            for doc, meta, dist in zip(docs, metadatas, distances):
                retrieved_chunks.append({
                    "text": doc,
                    "source": meta.get("source", "unknown"),
                    "chunk_index": meta.get("chunk_index", -1),
                    "distance": dist
                })
                
        return retrieved_chunks
