import os

from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoImageProcessor


class ImageDataset(Dataset):
    def __init__(
        self,
        processor=None,
        img_paths: list[str] = None,
        img_ids: list[str] = None,
    ):
        self.processor = processor
        self.img_paths = img_paths
        self.img_ids = img_ids

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index: int):
        img = Image.open(self.img_paths[index]).convert("RGB")
        if self.processor is not None:
            img = self.processor(img)
        return img, self.img_ids[index]


class MP16Dataset(Dataset):
    def __init__(
        self,
        df,
        img_col: str = "IMG_ID",
        id_col: str = "IMG_ID",
        img_base_path: str = "",
        model_name: str = "",
    ):
        self.df = df
        self.img_col = img_col
        self.id_col = id_col
        self.img_base_path = img_base_path
        self.processor = AutoImageProcessor.from_pretrained(model_name) if model_name else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df[index]
        path = row[self.img_col].item()
        path = path if ".jpg" in path else f"{path}.jpg"
        path = os.path.join(self.img_base_path, path)
        img_id = row[self.id_col].item()

        img = Image.open(path).convert("RGB")

        if self.processor:
            inputs = self.processor(images=img, return_tensors="pt")
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            return inputs, img_id

        return {"image": img, "size": img.size[::-1]}, img_id
