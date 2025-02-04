import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from pytorch_lightning import LightningDataModule
from PIL import Image
from kagglehub import kagglehub
import typer
from typing import Tuple, List, Optional


def load_data(dataset_path: Optional[str] = None) -> Tuple[pd.DataFrame, transforms.Compose, List[str], str]:
    """
    Loads the sea animals dataset. Downloads it using kagglehub if not already present.
    """
    # If no custom path is provided, use the default path
    if dataset_path is None:
        dataset_path = os.getenv(
            "DATASET_PATH",
            os.path.join(
                os.path.expanduser("~"), ".cache/kagglehub/datasets/vencerlanz09/sea-animals-image-dataste"
            )
        )

    # Check if dataset exists, otherwise download it
    if os.path.exists(dataset_path):
        print(f"Dataset found at {dataset_path}.")
    else:
        print("Starting download")
        try:
            path = kagglehub.dataset_download("vencerlanz09/sea-animals-image-dataste")
            print(f"Dataset downloaded at {path}")
            dataset_path = path  # Ensure the returned path is used
        except Exception as e:
            raise RuntimeError("Dataset download failed") from e
    # Gather file paths and labels

    classes = []
    paths = []
    for dirname, _, filenames in os.walk(dataset_path):
        for filename in filenames:
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                classes.append(dirname.split('/')[-1])
                paths.append(os.path.join(dirname, filename))

    print(f"Found {len(paths)} image files.")

    # Create mappings for classes
    class_names = sorted(set(classes))
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

    # Build DataFrame
    data = pd.DataFrame({'path': paths, 'class': classes})
    data['label'] = data['class'].map(class_to_idx)

    # Shuffle the data
    data = data.sample(frac=1).reset_index(drop=True)

    # Define transformations
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return data, transform, class_names, dataset_path


class CustomDataset(Dataset):
    def __init__(self, data: pd.DataFrame, transform: Optional[transforms.Compose] = None):
        """
        Custom dataset for handling image paths and labels.
        """
        self.data = data
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        row = self.data.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        label = row['label']

        if self.transform:
            img = self.transform(img)

        return img, label


class ImageDataModule(LightningDataModule):
    def __init__(self, data: pd.DataFrame, transform: transforms.Compose, batch_size: int = 32):
        super().__init__()
        self.data = data
        self.transform = transform
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Splits the dataset into training and validation sets.
        """
        dataset = CustomDataset(self.data, self.transform)
        dataset_size = len(dataset)
        train_size = int(0.6 * dataset_size)
        val_size = dataset_size - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self) -> DataLoader:
        """
        Returns DataLoader for training dataset.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=15)

    def val_dataloader(self) -> DataLoader:
        """
        Returns DataLoader for validation dataset.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=15)


def main() -> None:
    data, transform, class_names, dataset_path = load_data()

    # Create the data module
    data_module = ImageDataModule(data, transform, batch_size=32)

    # Set up the data loaders
    data_module.setup()

    # Print some statistics
    print(f"Number of training samples: {len(data_module.train_dataset)}")
    print(f"Number of validation samples: {len(data_module.val_dataset)}")
    print(f"Datasetpath: {dataset_path}")
    print(f"Datadimensions: {data.shape[1]}")
    print(f"Classes: {class_names}")

# Example Usage
if __name__ == "__main__":
    # Load the dataset and transform
    typer.run(main)
