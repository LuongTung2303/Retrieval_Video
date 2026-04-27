import torch
import clip
from PIL import Image

class CLIPEncoder:
    def __init__(self, model_name="ViT-L/14@336px"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # clip.load sẽ tự động tải weights và bộ preprocess tương ứng với 336px
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

    def get_vector(self, image_path):
        image = self.preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            vector = self.model.encode_image(image)
            vector = vector / vector.norm(dim=-1, keepdim=True)
        return vector.cpu().numpy().flatten().tolist()