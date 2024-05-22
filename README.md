<div align="center">

<h1>GNNInterpreter: A Probabilistic Generative Model-Level Explanation for Graph Neural Networks</h1>

[Xiaoqi Wang](https://scholar.google.com/citations?user=i__pLDEAAAAJ&hl=en&oi=sra)<sup>1</sup>, &nbsp;
[Han-Wei Shen](https://scholar.google.com/citations?user=95Z6-isAAAAJ&hl=en)<sup>1</sup>, &nbsp;


<sup>1</sup>[The Ohio State University](), &nbsp;

ICLR 2023

</div>

## 📖 Introduction
Recently, Graph Neural Networks (GNNs) have significantly advanced the performance of machine learning 
tasks on graphs. However, this technological breakthrough makes people wonder: how does a GNN make such
decisions, and can we trust its prediction with high confidence? When it comes to some critical fields,
such as biomedicine, where making wrong decisions can have severe consequences, it is crucial to interpret
the inner working mechanisms of GNNs before applying them. In this paper, we propose a model-agnostic 
model-level explanation method for different GNNs that follow the message passing scheme, GNNInterpreter, 
to explain the high-level decision-making process of the GNN model. More specifically, GNNInterpreter learns 
a probabilistic generative graph distribution that produces the most discriminative graph pattern the GNN 
tries to detect when making a certain prediction by optimizing a novel objective function specifically
designed for the model-level explanation for GNNs. Compared to existing works, GNNInterpreter is more flexible 
and computationally efficient in generating explanation graphs with different types of node and edge 
features, without introducing another blackbox or requiring manually specified domain-specific rules. 
In addition, the experimental studies conducted on four different datasets demonstrate that the explanation 
graphs generated by GNNInterpreter match the desired graph pattern if the model is ideal; otherwise, potential 
model pitfalls can be revealed by the explanation.

Paper: https://openreview.net/forum?id=rqq6Dh8t4d

## 🔥 How to use

### Notebooks
* `model_explanation_cyclicity.ipynb` contains the demo for the Cyclicity dataset experiment in the paper.
* `model_explanation_motif.ipynb` contains the demo for the Motif dataset experiment in the paper.
* `model_explanation_MUTAG.ipynb` contains the demo for the MUTAG dataset experiment in the paper.
* `model_explanation_Shape.ipynb` contains the demo for the Shape dataset experiment in the paper.

### Model Checkpoints
* You can find the GNN classifier checkpoints in the `ckpts` folder.
* See `model_training.ipynb` for how to load the model checkpoints.

### Datasets
* Here's the [link](https://drive.google.com/file/d/1vTmRR-nbo5SOQ_IwltManUzkq_4PNgNF/view?usp=sharing) for downloading the processed datasets.
* After downloading the datasets zip, please `unzip` it in the root folder.

### Environment
Codes in this repo have been tested on `python3.10` + `pytorch2.0` + `pyg2.3`.

To reproduce the exact python environment, please run:
```bash
conda create -n gnninterpreter poetry jupyter
conda activate gnninterpreter
poetry install
ipython kernel install --user --name=gnninterpreter --display-name="GNNInterpreter"
```

Note: In case poetry fails to install the dependencies, you can manually install them using `pip`:
```bash
pip install -r requirements.txt
````

## 🖼️ Demo
![demo](figures/demo.png)

##  Subsequent Work
Beyond the model-level explanation method explored in this work, we further extend the idea to explaining the decision boundaries of GNNs. We propose another model-level explainability method called GNNBoundary. Please check this [repository](https://github.com/yolandalalala/GNNBoundary) for more details.


## 🖊️ Citation
If you used our code or find our work useful in your research, please consider citing:
```
@inproceedings{
wang2023gnninterpreter,
title={{GNNI}nterpreter: A Probabilistic Generative Model-Level Explanation for Graph Neural Networks},
author={Xiaoqi Wang and Han Wei Shen},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=rqq6Dh8t4d}
}
```

## 🙏 Acknowledgement
The work  was supported in part by  the US Department of Energy SciDAC program DE-SC0021360,
National Science Foundation Division of Information and Intelligent Systems IIS-1955764,
and National Science Foundation Office of Advanced Cyberinfrastructure OAC-2112606.
