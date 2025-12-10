import os
import glob
import json
from pathlib import Path
from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions
from chonkie import Pipeline

class RAGIngestor:
    def __init__(self, data_dir: str, profession: str, chunk_size: int = 512, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        chunk_size: Number of TOKENS (not characters) for Chonkie.
        """
        self.data_dir = Path(data_dir)
        self.profession = profession
        self.chunk_size = chunk_size
        
        self.docs_dir = self.data_dir / "rag" / "documents" / profession
        self.db_dir = self.data_dir / "rag" / "vector_database" / profession
        
        # Ensure directories exist
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=str(self.db_dir))
        
        # Use sentence-transformers for embeddings
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        
        self.collection = self.client.get_or_create_collection(
            name=f"rag_{self._sanitize_name(profession)}",
            embedding_function=self.embedding_func
        )

    def _sanitize_name(self, name: str) -> str:
        return "".join(c if c.isalnum() else "_" for c in name)

    def ingest(self):
        """Read files, chunk them using Chonkie Pipeline, and store in DB."""
        print(f"Initializing Chonkie Pipeline for {self.profession}...")
        
        # Build pipeline
        # Using process_with("markdown") to handle tables and code blocks correctly
        pipeline = (Pipeline()
            .fetch_from("file", dir=str(self.docs_dir), ext=[".md", ".txt"])
            .process_with("markdown")
            .chunk_with("recursive", chunk_size=self.chunk_size, min_characters_per_chunk=24)
            .refine_with("overlap", method="prefix", context_size=50)
            .refine_with("overlap", method="suffix", context_size=50)
        )
        
        try:
            print(f"Running pipeline on {self.docs_dir}...")
            docs = pipeline.run()
        except Exception as e:
            print(f"Error running pipeline: {e}")
            return

        if not docs:
            print(f"No documents processed from {self.docs_dir}")
            return

        print(f"Processed {len(docs)} documents.")
        
        all_chunks = []
        all_ids = []
        all_metadatas = []
        
        doc_count = 0
        for doc in docs:
            # Try to get filename from metadata
            # Chonkie file fetcher usually puts path/filename in metadata
            source_path = getattr(doc, 'metadata', {}).get('path', 'unknown')
            if source_path == 'unknown':
                 # Fallback if metadata structure is different
                 source_path = getattr(doc, 'path', f"doc_{doc_count}")
            
            filename = Path(source_path).name
            
            for i, chunk in enumerate(doc.chunks):
                all_chunks.append(chunk.text)
                all_ids.append(f"{filename}_{i}")
                all_metadatas.append({"source": filename, "chunk_index": i})
            
            doc_count += 1

        if all_chunks:
            # Save chunks to JSON for inspection
            chunks_dump_path = self.docs_dir / "chunks_debug.json"
            debug_data = []
            for i in range(len(all_chunks)):
                debug_data.append({
                    "id": all_ids[i],
                    "text": all_chunks[i],
                    "metadata": all_metadatas[i]
                })
            
            with open(chunks_dump_path, 'w', encoding='utf-8') as f:
                json.dump(debug_data, f, indent=2, ensure_ascii=False)
            print(f"Saved chunks for inspection to {chunks_dump_path}")

            # Add to collection (upsert to overwrite existing)
            # Process in batches to avoid hitting limits if any
            batch_size = 100
            total_chunks = len(all_chunks)
            print(f"Ingesting {total_chunks} chunks into Vector DB...")
            
            for i in range(0, total_chunks, batch_size):
                end = min(i + batch_size, total_chunks)
                self.collection.upsert(
                    documents=all_chunks[i:end],
                    ids=all_ids[i:end],
                    metadatas=all_metadatas[i:end]
                )
            print(f"Successfully ingested {doc_count} documents ({total_chunks} chunks).")
        else:
            print("No valid text content found to ingest.")

if __name__ == "__main__":
    # Example usage
    # ingestor = RAGIngestor("path/to/data", "Profession Name")
    # ingestor.ingest()
    pass
