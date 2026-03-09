import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class DocumentIndexer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            logger.error(f"Failed to load sentence-transformers model: {e}")
            self.model = None
        
        self.index = None
        self.chunks = []
        
    def build_index(self, extracted_text_json, chunk_size=1000):
        """
        Builds a FAISS index from the extracted document JSON strings.
        """
        if not self.model:
            return
            
        try:
            data = json.loads(extracted_text_json)
            combined_text = "\n\n".join([f"--- Page {item['page_number']} ---\n{item['text']}" for item in data])
        except Exception:
            combined_text = extracted_text_json
            
        # Split into smaller chunks for semantic search
        self.chunks = []
        for i in range(0, len(combined_text), chunk_size):
            self.chunks.append(combined_text[i:i+chunk_size])
            
        if not self.chunks:
            return
            
        try:
            embeddings = self.model.encode(self.chunks)
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(np.array(embeddings).astype('float32'))
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}")
            self.index = None
            
    def search(self, query, top_k=3):
        """
        Search for the most relevant chunks related to the query.
        """
        if not self.index or not self.model or not self.chunks:
            return []
            
        try:
            query_vector = self.model.encode([query])
            distances, indices = self.index.search(np.array(query_vector).astype('float32'), top_k)
            
            results = []
            for idx in indices[0]:
                if idx < len(self.chunks) and idx >= 0:
                    results.append(self.chunks[idx])
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

# Singleton instance to avoid reloading the model
_indexer_instance = None

def get_indexer():
    global _indexer_instance
    if _indexer_instance is None:
        _indexer_instance = DocumentIndexer()
    return _indexer_instance
