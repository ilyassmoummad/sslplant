import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="Device to use for training (e.g., 'cuda' or 'cpu')",
)

parser.add_argument(
    "--data_dir",
    type=str,
    default="~/.cache/openml/org/openml/www/datasets/   ",
    help="Path to the evaluation dataset directory",
)
parser.add_argument(
    "--n_workers",
    type=int,
    default=16,
    help="Number of workers for the data loader",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=256,
    help="Batch size for training/evaluation",
)
parser.add_argument(
    "--resize_size",
    type=int,
    default=256,
    help="Image resize dimension before cropping",
)
parser.add_argument(
    "--crop_size",
    type=int,
    default=224,
    help="Final crop size for input images",
)

parser.add_argument(
    "--ckpt",
    type=str,
    help="Path to model checkpoint weights",
)

parser.add_argument(
    "--evaldataset",
    type=str,
    help="Dataset used for evaluation (e.g., 'plantvillage', 'plantnet', 'medleaf', 'plantdoc')",
)

parser.add_argument(
    "--shots",
    type=int,
    default=5,
    help="Number of shots for few-shot learning",
)
parser.add_argument(
    "--fold",
    type=int,
    default=5,
    help="Number of folds for few-shot evaluation",
)

args = parser.parse_args()