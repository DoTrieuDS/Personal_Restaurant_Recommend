import pandas as pd
import numpy as np
import faiss
import pickle
import os
from typing import Dict, List, Tuple, Optional
import logging
from .data_config import DataConfig

class FAISSVectorDB:
    """
    FAISS Vector Database cho POI (Point of Interest)
    Quản lý embedding vectors và mapping với business_id, metadata
    """
    
    def __init__(self, embedding_dim: int = 1024):
        """
        Khởi tạo FAISS Vector Database
        
        Args:
            embedding_dim: Kích thước embedding vector (mặc định 1024 cho BGE-M3)
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.business_id_to_index = {}  # mapping business_id -> faiss_index
        self.index_to_business_id = {}  # mapping faiss_index -> business_id
        self.poi_metadata = {}  # mapping business_id -> metadata
        self.logger = logging.getLogger(__name__)
        
    def load_embeddings_from_parquet(self, embeddings_file: str) -> Tuple[np.ndarray, List[str]]:
        """
        Đọc embeddings từ file Parquet
        
        Args:
            embeddings_file: Đường dẫn đến file poi_embeddings_structured.parquet
            
        Returns:
            Tuple[embeddings_matrix, business_ids_list]
        """
        try:
            df = pd.read_parquet(embeddings_file)
            self.logger.info(f"Đã đọc {len(df)} embeddings từ {embeddings_file}")
            
            # Lấy danh sách business_id
            business_ids = df['business_id'].tolist()
            
            # Chuyển đổi embeddings từ object sang numpy array
            embeddings_list = []
            for idx, embedding in enumerate(df['embedding']):
                if isinstance(embedding, (list, np.ndarray)):
                    embeddings_list.append(np.array(embedding, dtype=np.float32))
                else:
                    # Nếu embedding được lưu dưới dạng string hoặc format khác
                    try:
                        if isinstance(embedding, str):
                            # Parse string representation of array
                            embedding = eval(embedding)
                        embeddings_list.append(np.array(embedding, dtype=np.float32))
                    except Exception as e:
                        self.logger.warning(f"Không thể parse embedding tại index {idx}: {e}")
                        continue
            
            # Chuyển thành matrix
            embeddings_matrix = np.vstack(embeddings_list)
            self.logger.info(f"Tạo embedding matrix với shape: {embeddings_matrix.shape}")
            
            return embeddings_matrix, business_ids
            
        except Exception as e:
            self.logger.error(f"Lỗi khi đọc embeddings: {e}")
            raise
    
    def load_poi_metadata(self, metadata_file: str) -> Dict:
        """
        Đọc metadata POI từ file Parquet
        
        Args:
            metadata_file: Đường dẫn đến file pois_with_cleaned_descriptions.parquet
            
        Returns:
            Dictionary mapping business_id -> metadata
        """
        try:
            df = pd.read_parquet(metadata_file)
            self.logger.info(f"Đã đọc metadata cho {len(df)} POIs từ {metadata_file}")
            
            # Tạo dictionary metadata
            metadata_dict = {}
            for _, row in df.iterrows():
                business_id = row['business_id']
                metadata_dict[business_id] = {
                    'name': row.get('name', ''),
                    'city': row.get('city', ''),
                    'state': row.get('state', ''),
                    'stars': row.get('stars', 0.0),
                    'review_count': row.get('review_count', 0),
                    'categories': row.get('categories', ''),
                    'latitude': row.get('latitude', 0.0),
                    'longitude': row.get('longitude', 0.0),
                    'poi_type': row.get('poi_type', ''),
                    'description': row.get('description', ''),
                    'cleaned_description': row.get('cleaned_description', '')
                }
            
            return metadata_dict
            
        except Exception as e:
            self.logger.error(f"Lỗi khi đọc metadata: {e}")
            raise
    
    def build_faiss_index(self, embeddings: np.ndarray, index_type: str = "IndexFlatIP") -> faiss.Index:
        """
        Xây dựng FAISS index từ embeddings
        
        Args:
            embeddings: Matrix embeddings (n_samples, embedding_dim)
            index_type: Loại FAISS index ("IndexFlatIP", "IndexFlatL2", "IndexIVFFlat")
            
        Returns:
            FAISS index đã được train
        """
        try:
            n_samples, dim = embeddings.shape
            self.logger.info(f"Xây dựng FAISS index {index_type} cho {n_samples} vectors, dim={dim}")
            
            # Normalize embeddings cho Inner Product search
            faiss.normalize_L2(embeddings)
            
            if index_type == "IndexFlatIP":
                # Flat index với Inner Product (cho cosine similarity)
                index = faiss.IndexFlatIP(dim)
            elif index_type == "IndexFlatL2":
                # Flat index với L2 distance
                index = faiss.IndexFlatL2(dim)
            elif index_type == "IndexIVFFlat":
                # IVF index cho large datasets
                nlist = min(int(np.sqrt(n_samples)), 1000)  # số clusters
                quantizer = faiss.IndexFlatIP(dim)
                index = faiss.IndexIVFFlat(quantizer, dim, nlist)
                index.train(embeddings)
            else:
                raise ValueError(f"Unsupported index type: {index_type}")
            
            # Thêm vectors vào index
            index.add(embeddings)
            
            self.logger.info(f"Đã xây dựng thành công FAISS index với {index.ntotal} vectors")
            return index
            
        except Exception as e:
            self.logger.error(f"Lỗi khi xây dựng FAISS index: {e}")
            raise
    
    def create_mapping(self, business_ids: List[str]):
        """
        Tạo mapping giữa FAISS index và business_id
        
        Args:
            business_ids: Danh sách business_id theo thứ tự trong FAISS index
        """
        self.business_id_to_index = {bid: idx for idx, bid in enumerate(business_ids)}
        self.index_to_business_id = {idx: bid for idx, bid in enumerate(business_ids)}
        self.logger.info(f"Đã tạo mapping cho {len(business_ids)} business IDs")
    
    def build_database(self, 
                      embeddings_file: str, 
                      metadata_file: str, 
                      index_type: str = "IndexFlatIP"):
        """
        Xây dựng hoàn chỉnh Vector Database
        
        Args:
            embeddings_file: Đường dẫn file poi_embeddings_structured.parquet
            metadata_file: Đường dẫn file pois_with_cleaned_descriptions.parquet
            index_type: Loại FAISS index
        """
        try:
            # 1. Đọc embeddings
            embeddings, business_ids = self.load_embeddings_from_parquet(embeddings_file)
            
            # 2. Đọc metadata
            self.poi_metadata = self.load_poi_metadata(metadata_file)
            
            # 3. Xây dựng FAISS index
            self.index = self.build_faiss_index(embeddings, index_type)
            
            # 4. Tạo mapping
            self.create_mapping(business_ids)
            
            self.logger.info("Hoàn thành xây dựng Vector Database!")
            
        except Exception as e:
            self.logger.error(f"Lỗi khi xây dựng database: {e}")
            raise
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Dict]:
        """
        Tìm kiếm k POIs gần nhất với query vector
        
        Args:
            query_vector: Vector query (1, embedding_dim)
            k: Số lượng kết quả trả về
            
        Returns:
            List các POI với metadata và similarity score
        """
        if self.index is None:
            raise ValueError("FAISS index chưa được xây dựng!")
        
        # Normalize query vector
        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_vector)
        
        # Tìm kiếm
        scores, indices = self.index.search(query_vector, k)
        
        # Trả về kết quả với metadata
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # Không tìm thấy
                continue
                
            business_id = self.index_to_business_id[idx]
            metadata = self.poi_metadata.get(business_id, {})
            
            results.append({
                'rank': i + 1,
                'business_id': business_id,
                'similarity_score': float(score),
                'metadata': metadata
            })
        
        return results
    
    def get_poi_by_business_id(self, business_id: str) -> Optional[Dict]:
        """
        Lấy thông tin POI theo business_id
        """
        return self.poi_metadata.get(business_id)
    
    def save_database(self, save_dir: str = None):
        """
        Lưu FAISS database và mappings
        """
        if save_dir is None:
            save_dir = DataConfig.ALL_POIS_INDEX_DIR
            
        os.makedirs(save_dir, exist_ok=True)
        
        # Lưu FAISS index
        faiss.write_index(self.index, os.path.join(save_dir, "faiss_index.bin"))
        
        # Lưu mappings và metadata
        with open(os.path.join(save_dir, "mappings.pkl"), "wb") as f:
            pickle.dump({
                'business_id_to_index': self.business_id_to_index,
                'index_to_business_id': self.index_to_business_id,
                'poi_metadata': self.poi_metadata
            }, f)
        
        self.logger.info(f"Đã lưu database tại: {save_dir}")
    
    def load_database(self, save_dir: str = None):
        """
        Load FAISS database từ disk
        """
        if save_dir is None:
            save_dir = DataConfig.ALL_POIS_INDEX_DIR
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(save_dir, "faiss_index.bin"))
        
        # Load mappings và metadata
        with open(os.path.join(save_dir, "mappings.pkl"), "rb") as f:
            data = pickle.load(f)
            self.business_id_to_index = data['business_id_to_index']
            self.index_to_business_id = data['index_to_business_id']
            self.poi_metadata = data['poi_metadata']
        
        self.logger.info(f"Đã load database từ: {save_dir}")

    def filter_restaurants_and_build(self, 
                                    embeddings_file: str, 
                                    metadata_file: str, 
                                    save_dir: str = None,
                                    index_type: str = "IndexFlatIP"):
        """
        Filter chỉ restaurants và build restaurant-only database
        
        Args:
            embeddings_file: Đường dẫn file poi_embeddings_structured.parquet
            metadata_file: Đường dẫn file pois_with_cleaned_descriptions.parquet
            save_dir: Thư mục để lưu restaurant-only database
            index_type: Loại FAISS index
        """
        try:
            # 1. Đọc embeddings và metadata
            all_embeddings, all_business_ids = self.load_embeddings_from_parquet(embeddings_file)
            all_metadata = self.load_poi_metadata(metadata_file)
            
            # 2. Filter chỉ restaurants
            restaurant_indices = []
            restaurant_business_ids = []
            restaurant_metadata = {}
            
            for idx, business_id in enumerate(all_business_ids):
                metadata = all_metadata.get(business_id, {})
                
                # Check if it's a restaurant
                categories = metadata.get('categories', '').lower()
                poi_type = metadata.get('poi_type', '').lower()
                
                is_restaurant = (
                    'restaurant' in categories or 
                    'food' in categories or
                    'cafe' in categories or
                    poi_type == 'restaurant' or
                    'dining' in categories or
                    'bar' in categories or
                    'bakery' in categories
                )
                
                if is_restaurant:
                    restaurant_indices.append(idx)
                    restaurant_business_ids.append(business_id)
                    restaurant_metadata[business_id] = metadata
            
            # 3. Extract restaurant embeddings
            restaurant_embeddings = all_embeddings[restaurant_indices]
            
            self.logger.info(f"Filtered {len(restaurant_embeddings)} restaurants from {len(all_embeddings)} total POIs")
            
            # 4. Build FAISS index with restaurant embeddings only
            self.index = self.build_faiss_index(restaurant_embeddings, index_type)
            
            # 5. Create new mapping
            self.create_mapping(restaurant_business_ids)
            
            # 6. Update metadata
            self.poi_metadata = restaurant_metadata
            
            # 7. Save restaurant-only database
            if save_dir is None:
                from .data_config import DataConfig
                save_dir = DataConfig.RESTAURANT_INDEX_DIR
            
            os.makedirs(save_dir, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, os.path.join(save_dir, "faiss_index.bin"))
            
            # Save mappings và metadata
            with open(os.path.join(save_dir, "mappings.pkl"), "wb") as f:
                pickle.dump({
                    'business_id_to_index': self.business_id_to_index,
                    'index_to_business_id': self.index_to_business_id,
                    'poi_metadata': self.poi_metadata
                }, f)
            
            self.logger.info(f"Restaurant-only database đã được lưu tại: {save_dir}")
            self.logger.info(f"Total restaurants: {len(restaurant_metadata)}")
            
            # Return stats
            return {
                'total_restaurants': len(restaurant_metadata),
                'original_total': len(all_embeddings),
                'filter_ratio': len(restaurant_metadata) / len(all_embeddings),
                'save_directory': save_dir
            }
            
        except Exception as e:
            self.logger.error(f"Lỗi khi filter restaurants: {e}")
            raise

def main():
    """
    Hàm main để test và xây dựng database
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Đường dẫn files
    embeddings_file = "modules/recommendation/data/poi_embeddings_structured.parquet"
    metadata_file = "modules/recommendation/data/pois_with_cleaned_descriptions.parquet"
    save_dir = "modules/recommendation/faiss_db"
    
    # Khởi tạo và xây dựng database
    vector_db = FAISSVectorDB(embedding_dim=1024)  # BGE-M3 embedding dimension
    
    try:
        # Xây dựng database
        vector_db.build_database(embeddings_file, metadata_file, index_type="IndexFlatIP")
        
        # Lưu database
        vector_db.save_database(save_dir)
        
        # Test search với random vector
        print("\n=== Test tìm kiếm ===")
        random_vector = np.random.rand(1, 1024).astype(np.float32)
        results = vector_db.search(random_vector, k=5)
        
        for result in results:
            print(f"Rank {result['rank']}: {result['metadata']['name']} - Score: {result['similarity_score']:.4f}")
            
    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    main()