from pymilvus import MilvusClient, DataType

class HybridMilvusDatabase:
    def __init__(self, uri, token):
        # Kết nối tới Zilliz Cloud
        self.client = MilvusClient(uri=uri, token=token)
        print("Đã kết nối tới Zilliz Cloud thành công!")

    def create_hybrid_collection(self, collection_name="hybrid_video_search"):
        # Kiểm tra nếu collection đã tồn tại thì bỏ qua
        if self.client.has_collection(collection_name):
            print(f"Collection '{collection_name}' đã tồn tại.")
            return

        print("Đang tạo Schema cho Hybrid Search...")
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)

        # Thêm các trường dữ liệu
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="video_id", datatype=DataType.VARCHAR, max_length=200)
        schema.add_field(field_name="frame_timestamp", datatype=DataType.FLOAT)
        # Khai báo 2 cột vector 768 chiều cho CLIP và BEiT-3
        schema.add_field(field_name="clip_vector", datatype=DataType.FLOAT_VECTOR, dim=768)
        schema.add_field(field_name="beit3_vector", datatype=DataType.FLOAT_VECTOR, dim=768)

        # Cấu hình Index để tìm kiếm nhanh
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="clip_vector", index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index(field_name="beit3_vector", index_type="AUTOINDEX", metric_type="COSINE")

        # Khởi tạo Collection trên Zilliz
        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )
        print(f"Khởi tạo thành công Collection: {collection_name}!")
        
    # --- PHẦN BẠN CẦN THÊM VÀO ---
    def insert_data(self, collection_name, data):
        """
        data: list các dictionary có các key khớp với schema
        """
        return self.client.insert(collection_name=collection_name, data=data)