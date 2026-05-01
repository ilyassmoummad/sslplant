import torch
import torch.nn.functional as F
import numpy as np
from numpy import linalg as LA
from get_model import get_encoder
from get_data import get_loader
import random
from args import args


def extract_features(encoder, dataloader, args):
    encoder.eval()
    features, targets = [], []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(args.device)

            if args.model == 'bioclip':
                feats = encoder.encode_image(imgs)
            else:
                feats = encoder(imgs)

            if 'dinov2' in args.model:
                feats = feats['x_norm_clstoken']

            feats = F.normalize(feats, dim=-1)
            features.append(feats.cpu().numpy())
            targets.append(labels.numpy())

    return np.vstack(features), np.hstack(targets)


def few_shot_split(features, targets, kshot, nfold):
    select = {}
    for f, t in zip(features, targets):
        select.setdefault(t, {'feature': []})
        select[t]['feature'].append(f)
    
    n_class = len(select)
    train, train_labels, test, test_labels = [], [], [], []
    for cls, v in select.items():
        feats = v['feature']
        random.seed(nfold)
        random.shuffle(feats)
        train.extend(feats[:kshot])
        train_labels.extend([cls]*kshot)
        test.extend(feats[kshot:])
        test_labels.extend([cls]*(len(feats)-kshot))
    
    return np.array(train), np.array(train_labels), np.array(test), np.array(test_labels), n_class


def normalize_classwise(x, x_mean):
    x = x - x_mean
    x = x / LA.norm(x, axis=1, keepdims=True)
    return x


def nearest_centroid_acc(train_feats, train_labels, test_feats, test_labels, kshot, n_class):
    mean_train = train_feats.mean(axis=0)
    train_feats = normalize_classwise(train_feats, mean_train)
    test_feats = normalize_classwise(test_feats, mean_train)
    centers = train_feats.reshape(n_class, kshot, -1).mean(axis=1)
    
    distances = LA.norm(centers[:, None, :] - test_feats, axis=-1)
    preds = train_labels[::kshot][np.argmin(distances, axis=0)]
    
    return (preds == test_labels).mean()


def few_shot_evaluation(encoder, dataloader, args):
    features, targets = extract_features(encoder, dataloader, args)
    
    results = {}
    acc_list = []

    for fold in range(args.fold):
        train, train_labels, test, test_labels, n_class = few_shot_split(features, targets, args.shots, fold)
        acc = nearest_centroid_acc(train, train_labels, test, test_labels, args.shots, n_class)
        acc_list.append(acc)
    results[args.shots] = (np.mean(acc_list), np.std(acc_list))
    
    return results


def main():
    encoder = get_encoder(args)
    loader = get_loader(args)
    results = few_shot_evaluation(encoder, loader, args)

    for kshot, (mean_acc, std_acc) in results.items():
        print(f"{kshot}-shot accuracy: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")


if __name__ == "__main__":
    main()