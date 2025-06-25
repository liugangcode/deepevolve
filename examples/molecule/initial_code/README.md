Graph Rationalization with Environment-based Augmentations
====

This is the source code for the KDD'22 paper:

[Graph Rationalization with Environment-based Augmentations](https://arxiv.org/pdf/2206.02886.pdf)

by [Gang Liu](https://liugangcode.github.io/) ([gliu7@nd.edu](mailto:gliu7@nd.edu)), [Tong Zhao](https://tzhao.io/), Jiaxin Xu, [Tengfei Luo](https://monsterlab.nd.edu/), [Meng Jiang](http://www.meng-jiang.com/)

## Requirements

This code package was developed and tested with Python 3.9.9 and PyTorch 1.10.1. All dependencies specified in the ```requirements.txt``` file. The packages can be installed by
```
pip install -r requirements.txt
```

## Usage

Following are the commands to run experiments on polymer or molecule datasets using default settings.

```
# OGBG-HIV for example
python main_pyg.py --dataset ogbg-molhiv --by_default

# Polymer Oxygen Permeability
python main_pyg.py --dataset plym-o2_prop --by_default
```

## Datasets

We provide the oxygen permeability dataset (.csv) for polymer graph regression. It can be found in the ``` data/'name'/raw ``` folder. 

> Update March 26, 2025: We delegated the polymer datasets for GlassTemp, MeltingTemp, and PolyDensit as requested by the NIMS Materials Database, MatNavi.

Binary classification tasks for the OGBG dataset (i.e., HIV, ToxCast, Tox21, BBBP, BACE, ClinTox and SIDER) can be directedly implemented using commands such as ``` --dataset ogbg-molhiv ``` following the [instructions](https://github.com/snap-stanford/ogb/tree/master/examples/graphproppred/mol) of the official OGBG dataset implementations.

## Reference

If you find this repository useful in your research, please cite our paper:

```bibtex
@inproceedings{liu2022graph,
  title={Graph Rationalization with Environment-based Augmentations},
  author={Liu, Gang and Zhao, Tong and Xu, Jiaxin and Luo, Tengfei and Jiang, Meng},
  booktitle = {Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  publisher = {Association for Computing Machinery},
  pages = {1069–1078},
  numpages = {10},
  year={2022}
}
```
