"""
Script to create/update the RAG Vector Database.
Run this ONCE when your source documents change.
"""

import yaml
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from RAG.ingest import RAGIngestor

def main():
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "config.yaml"
    
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    rag_cfg = config.get("benchmarking_rag", {})
    if not rag_cfg:
        print("Error: 'benchmarking_rag' section missing in config.")
        return

    professions = rag_cfg.get("professions", [])
    if not professions:
        print("No professions defined in benchmarking_rag.professions")
        return

    # Get RAG parameters
    rag_params = rag_cfg.get("rag_parameters", {})
    chunk_size = rag_params.get("chunk_size", 512)
    embedding_model = rag_params.get("embedding_model", "all-MiniLM-L6-v2")
    
    # Data directory (assuming standard structure relative to project root)
    # config has raw_data_dir, we need the parent of that usually, or just use the path logic from ingest
    # Ingestor expects 'data_dir' which contains 'rag/documents/...'
    # The config['raw_data_dir'] is .../data/raw
    # So data_dir should be .../data
    
    raw_data_path = Path(config.get("raw_data_dir", "data/raw"))
    # If raw_data_path is absolute, this works. If relative, we might need to resolve it against project_root
    if not raw_data_path.is_absolute():
        raw_data_path = project_root / raw_data_path
        
    data_dir = raw_data_path.parent
    
    print(f"RAG Database Creation")
    print(f"Data Directory: {data_dir}")
    print(f"Parameters: Chunk Size={chunk_size}, Model={embedding_model}")
    print("-" * 50)

    for profession in professions:
        print(f"\nProcessing profession: {profession}")
        try:
            ingestor = RAGIngestor(
                data_dir=str(data_dir),
                profession=profession,
                chunk_size=chunk_size,
                embedding_model=embedding_model
            )
            ingestor.ingest()
        except Exception as e:
            print(f"Error processing {profession}: {e}")

    print("\n" + "="*50)
    print("RAG Database Creation Complete!")

if __name__ == "__main__":
    main()
