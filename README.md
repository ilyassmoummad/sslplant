# Self-Supervised Learning of Plant Image Representations (arXiv 2026)

Authors: [Ilyass Moummad](https://ilyassmoummad.github.io/), [Kawtar Zaher](https://scholar.google.com/citations?user=I-BoVeAAAAAJ&hl=en), [Hervé Goëau](https://scholar.google.com/citations?user=zBZYEzAAAAAJ&hl=fr), [Jean-Christophe Lombardo](http://www-sop.inria.fr/members/Jean-Christophe.Lombardo/), [Pierre Bonnet](https://agents.cirad.fr/Pierre+BONNET), [Alexis Joly](https://sites.google.com/view/alexis-joly-inria/home)

[[`Paper`](https://arxiv.org/abs/2604.27538)][[`BibTeX`](#to-cite-this-work)]

---

PyTorch implementation of training self-supervised models on iNaturalist 2021 Plantae dataset (~1.1M images) using [SimDINOv2](https://github.com/RobinWu218/SimDINO).

<div align="center">
  <image src="assets/ssl.png" width="840px" />
  <p></p>
</div>


## Pretrained models
We provide checkpoints for both ViT-B and ViT-L pretrained on iNaturalist 2021 Plantae subset for 100 epochs following configs detailed in our [paper](https://arxiv.org/abs/2604.27538).

<table style="margin: auto">
  <thead>
    <tr>
      <th>model</th>
      <th># of<br />params</th>
      <th>PlantNet<br>(5-shot)</th>
      <th>PlantVillage<br>(5-shot)</th>
      <th>Med. Leaf<br>(5-shot)</th>
      <th>PlantDoc<br>(5-shot)</th>
      <th>download</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-B/16</td>
      <td align="right">86 M</td>
      <td align="right">87.9 &plusmn; 0.7</td>
      <td align="right">85.4 &plusmn; 0.5</td>
      <td align="right">97.8 &plusmn; 0.7</td>
      <td align="right">47.8 &plusmn; 1.3</td>
      <td><a href="https://drive.google.com/file/d/1lpEfov-QmAWer3B_TC-icmb4DavOpisD/view?usp=sharing">ckpt</a></td>
    </tr>
    <tr>
      <td>ViT-L/16</td>
      <td align="right">300 M</td>
      <td align="right">88.2 &plusmn; 1.6</td>
      <td align="right">86.6 &plusmn; 0.1</td>
      <td align="right">98.5 &plusmn; 0.5</td>
      <td align="right">49.0 &plusmn; 1.1</td>
      <td><a href="https://drive.google.com/file/d/1OTs0XETGDwIs4Hvs9jukvFAifSlcH0xl/view?usp=sharing">ckpt</a></td>
    </tr>
  </tbody>
</table>

Note: We are currently working on scaling up our approach on [Pl@ntNet](https://plantnet.org/en/). Stay tuned for more model checkpoints in the future.


## Installation

Our implementation requires Python 3.11+ and PyTorch 2.4+ and some other packages. Note that the code has only been tested with the specified versions and also expects a Linux environment. To setup the dependencies, please install via:

```sh
pip install -r requirements.txt
```


## Data preparation

First, download [iNaturalist 2021](https://github.com/visipedia/inat_comp/tree/master/2021) dataset.

Then configure and run `python prepare_inatplant.py`. This script generates the required `.npy` metadata files.

For few-shot evaluation, download the following mini datasets from [**MetaAlbum**](https://meta-album.github.io/):

- [**PlantNet**](https://meta-album.github.io/datasets/PLT_NET.html)
- [**PlantVillage**](https://meta-album.github.io/datasets/PLT_VIL.html)
- [**Medicinal Leaf**](https://meta-album.github.io/datasets/MED_LF.html)
- [**PlantDoc**](https://meta-album.github.io/datasets/PLT_DOC.html)


## Training

### Training on iNaturalist 2021 Plantae subset (~1.1M images)

You can train SimDINOv2 on ViT-L/16 with a 8-GPU node (each with at least 40G memory):

```shell
torchrun --nnodes=1 --nproc_per_node=8 simdinov2/train/train.py \
    --config-file simdinov2/configs/simdino_config.yaml \
    --output-dir <PATH/TO/OUTPUT/DIR> \
    train.dataset_path=INatPlants:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

The training code saves the weights of the teacher in the `eval` folder every 10 epochs for evaluation. You can change the `student.arch` field in `simdino_config.yaml` to train other vision transformer models.

You can also use `submitit` if your environment happens to be a SLURM cluster:
```shell
python simdinov2/run/train/train.py \
    --nodes 1 \
    --config-file simdinov2/configs/simdino_config.yaml \
    --output-dir <PATH/TO/OUTPUT/DIR> \
    train.dataset_path=INatPlants:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```


## Evaluation

The teacher weights are regularly saved and can be evaluated using the script below. The evaluation code runs on a single GPU (no distributed setup required). 


### Few-shot classification on PlantNet Mini (MetaAlbum)

```shell
python simdinov2/eval/eval.py \
    --data_dir <PATH/TO/MetaAlbum/Datasets> \
    --dataset plantnet \
    --ckpt <PATH/TO/OUTPUT/DIR>/eval/training_X/teacher_checkpoint.pth \
    --shots 5 \
    --batch_size 256 \
    --n_workers 16 \
    --resize_size 256 \
    --crop_size 224
```
> Supported dataset options: `plantnet`, `plantvillage`, `medleaf`, `plantdoc`.


## To cite this work

If you find this project useful, please consider giving us a star and citation:
```
@misc{sslplant,
      title={Self-Supervised Learning of Plant Image Representations}, 
      author={Ilyass Moummad and Kawtar Zaher and Hervé Goëau and Jean-Christophe Lombardo and Pierre Bonnet and Alexis Joly},
      year={2026},
      eprint={2604.27538},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.27538}, 
}
```


## Acknowledgements

This project is largely built upon [SimDINO](https://github.com/RobinWu218/SimDINO) project.