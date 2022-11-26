# FireCLR-Wildfires
Official code for 🔥 Unsupervised Wildfire Change Detection based on Contrastive Learning 🔥. Work conducted at the <a href="https://frontierdevelopmentlab.org/fdl-2022">FDL USA 2022</a>.

<p align="center">
    
</p>

<p align="center">
  <a href="https://arxiv.org/abs/???">NeurIPS workshop paper [WIP]</a> •
  <a href="https://colab.research.google.com/github/spaceml-org/FireCLR-Wildfires/blob/master/notebooks/???">Quick Colab Example <img src="https://colab.research.google.com/assets/colab-badge.svg" height=16px> [WIP]</a>
</p>


---

## Unsupervised Wildfire Change Detection based on Contrastive Learning

<p align="center">
    <img src="_illustrations/example_wildfire.jpg" alt="Wildfire event example [WIP]" width="860px">
</p>


> **Abstract:** The accurate characterization of the severity of the wildfire event strongly contributes to the characterization of the fuel conditions in fire-prone areas, and provides valuable information for disaster response. The aim of this study is to develop an autonomous system built on top of high-resolution multispectral satellite imagery, with an advanced deep learning method for detecting burned area change. This work proposes an initial exploration of using an unsupervised model for feature extraction in wildfire scenarios. It is based on the contrastive learning technique _SimCLR_, which is trained to minimize the cosine distance between augmentations of images. The distance between encoded images can also be used for change detection. We propose changes to this method that allows it to be used for unsupervised burned area detection and following downstream tasks. We show that our proposed method outperforms the tested baseline approaches.

### Dataset

TO-DO

<p align="center">
    <img src="_illustrations/map_dataset.jpg" alt="Map of the events" width="500px">
</p>

### Code examples

**Install**

```bash
# prep environment
conda env create -f environment_droplet_mlenv.yml
```

**Inference**

```bash
# TODO
```

**Training**

```bash
# TODO
```

## Citation
If you find FireCLR useful in your research, please consider citing the following paper:
```
@inproceedings{fireclr2022} [TODO]
```

