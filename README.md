# Relation-based Frequency Adaptive Graph Neural Networks (RFA-GNN)


This is a PyTorch implementation of the RFA-GNN, and the code includes the following modules:

* Datasets (Cora, Citeseer, Pubmed, Texas, Cornell, Wisconsin, Film, Chameleon, Squirrel, Syn-Cora,  Syn-Relation, and ZINC)

* Training paradigm for node classification, graph classification, and graph regression tasks on 12 datasets

* Visualization

* Evaluation metrics 

  

## Main Requirements

* dgl==0.5.3
* networkx==2.5
* numpy==1.19.2
* matplotlib==3.1.1
* scikit-learn==0.24.1
* scipy==1.5.2
* torch==1.6.0



## Description

* train.py  
  * main() -- Train a new model for **node classification** task on the *Cora, Citeseer, Pubmed, Texas, Cornell, Wisconsin, Film, Chameleon, Squirrel, and Syn-Cora* datasets
  * accuracy() -- Test the learned model for **node classification** task on the *Cora, Citeseer, Pubmed, Texas, Cornell, Wisconsin, Film, Chameleon, Squirrel, and Syn-Cora* datasets
  * main_synthetic() -- Train a new model for **graph classification** task on the *Syn-Relation* dataset
  * evaluate_synthetic() -- Test the learned model for **graph classification** task on the *Syn-Relation* dataset
  * main_zinc() -- Train a new model for **graph regression** task on the *ZINC* datasets
  * evaluate_zinc() -- Test the learned model for **graph regression** task on the *ZINC* datasets
* dataset.py  
  
  * preprocess_data() -- Load data of selected dataset
* model_RFAGCN.py  
  
  * RFAGNN() -- model and loss
* utils.py  
  * evaluate_graph() -- Evaluate relation-learning performance with *the visualization of the learned relation graphs*



## Running the code

1. Install the required dependency packages

2. We use [DGL](https://www.dgl.ai/) to implement all the GCN models (and their modules) on 12 datasets. The three citation datasets (Cora, Citeseer, and Pubmed) are provided by the [DGL](https://www.dgl.ai/) library; the Syn-relation and  Syn-cora datasets are self-generated by the provided code `dataset.py`; the ZINC dataset and the remainding six heterophily datasets are downloaded from the [Google Drive](https://drive.google.com/file/d/1p3pMblv7eMRLtB4LERHwt8a50VvHp64n/view?usp=sharing).

3. To get the results on a specific *dataset*, run with proper hyperparameters

  ```
python train.py --dataset data_name
  ```

where the *data_name* is one of the 12 datasets (Cora, Citeseer, Pubmed, Texas, Cornell, Wisconsin, Film, Chameleon, Squirrel, Syn-relation,  Syn-cora, and Zinc). The model as well as the training log will be saved to the corresponding dir in **./log** for evaluation.

4. The evaluation the performance of three-level disentanglement performance, run

  ```
python utils.py
  ```



## Citation

If you find this project useful for your research, please use the following BibTeX entry.

```
@article{wu2023beyond,
  title={Beyond homophily and homogeneity assumption: Relation-based frequency adaptive graph neural networks},
  author={Wu, Lirong and Lin, Haitao and Hu, Bozhen and Tan, Cheng and Gao, Zhangyang and Liu, Zicheng and Li, Stan Z},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2023},
  publisher={IEEE}
}
```
