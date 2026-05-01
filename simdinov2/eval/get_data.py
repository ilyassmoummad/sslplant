import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


MODEL_STATS = {
    "mean": (0.429, 0.459, 0.328),
    "std": (0.221, 0.215, 0.221),
}


def get_transforms(args):

    return T.Compose([
        T.Resize((args.resize_size, args.resize_size), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(args.crop_size),
        T.ToTensor(),
        T.Normalize(mean=MODEL_STATS["mean"], std=MODEL_STATS["std"]),
    ])


class MetaAlbumDataset(Dataset):
    """
    Dataset for Meta-Album OpenML image datasets.

    Expected structure:
        dataset_root/
          ├── images/
          ├── labels.csv
          └── info.json
    """

    def __init__(self, dataset_root, transform=None):
        super().__init__()

        self.dataset_root = dataset_root
        self.images_dir = os.path.join(dataset_root, "images")
        self.transform = transform

        # Load labels
        labels_path = os.path.join(dataset_root, "labels.csv")
        self.labels_df = pd.read_csv(labels_path)

        # Create class-to-index mapping
        self.classes = sorted(self.labels_df["CATEGORY"].unique())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]

        # Image
        img_path = os.path.join(self.images_dir, row["FILE_NAME"])
        image = Image.open(img_path).convert("RGB")

        # Label
        label = self.class_to_idx[row["CATEGORY"]]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataset(args):
    """
    Select dataset based on args.dataset and args.data_dir.
    """

    DATASET_DIRS = {
        "plantnet": "44293/PLT_NET_Mini",
        "plantvillage": "44286/PLT_VIL_Mini",
        "medleaf": "44299/MED_LF_Mini",
        "plantdoc": "44303/PLT_DOC_Mini",
    }

    if args.dataset not in DATASET_DIRS:
        raise ValueError(f"Dataset '{args.dataset}' is not supported.")

    dataset_root = os.path.join(
        args.data_dir,
        DATASET_DIRS[args.dataset]
    )

    transform = get_transforms(args)

    return MetaAlbumDataset(dataset_root=dataset_root, transform=transform)


def get_loader(args):
    """
    Create DataLoader for evaluation.
    """

    dataset = get_dataset(args)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        pin_memory=True
    )

    return loader