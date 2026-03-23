from torch.utils.data import Dataset
import os
from PIL import Image
from transformers import AutoImageProcessor

from src import constant


class MP16Dataset(Dataset):
    def __init__(
        self,
        df,
        img_col: str = "IMG_ID",
        img_base_path: str = "",
    ):
        self.df = df
        self.img_col = img_col
        self.img_base_path = img_base_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        df_batch = self.df[index]
        path = df_batch[self.img_col].item()
        path = path if ".jpg" in path else f"{path}.jpg"
        path = os.path.join(self.img_base_path, path)

        img = Image.open(path).convert("RGB")
        # target_sizes expects (height, width)
        size = img.size[::-1]

        return {"image": img, "size": size}
    
def collate_fn(batch):
    processor = AutoImageProcessor.from_pretrained(constant.MASKFORMER_MODEL_NAME)
    images = [b["image"] for b in batch]
    sizes = [b["size"] for b in batch]
    inputs = processor(images=images, return_tensors="pt")
    return inputs, sizes, images
