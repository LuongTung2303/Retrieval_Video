import torch
from transformers import BeitImageProcessor, BeitForImageClassification # Hoặc dùng AutoModel
from PIL import Image

class BEiT3Encoder:
    def __init__(self, model_name="microsoft/beit-base-patch16-224-pt22k-ft22k"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Khởi tạo processor và model
        self.processor = BeitImageProcessor.from_pretrained(model_name)
        self.model = BeitForImageClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def get_vector(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.beit(**inputs)
            # Lấy vector ở pooler_output (thường là 768 chiều cho bản base)
            vector = outputs.pooler_output 
            # Normalize để tính Cosine Similarity chuẩn hơn
            vector = vector / vector.norm(dim=-1, keepdim=True)
            
        return vector.cpu().numpy().flatten().tolist()