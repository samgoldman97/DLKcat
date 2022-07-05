DLKcat
======

<p align="center">
  <img  src="doc/logo.png" width = "400">
</p>


Adding appropriate baselines
------------
The goal of this repository is to run a simple KNN baseline to see if in fact
something was learned by the deep learning model presented.  

This can be accomplished by installing conda as per the directions below and
simply running `Code/model/run_knn.py`. Note that two flags appear in the code
for both debugging and running hyperparameter grid sweep in parallel.

The resulting MAE and RMSE are 0.7029  and 1.061 respectively. This outperforms
the best deep learning model originally presented, which at epoch 18, achieved 
test MAE 0.7448 and RMSE 1.0911843.

Note that Figure 2B and 2C apply the trained model on both the training and 
validation set, so I have choosen not to replicate this.  

Introduction
------------

The **DLKcat** toolbox is a Matlab/Python package for prediction of
kcats and generation of the ecGEMs. The repo is divided into two parts:
`DeeplearningApproach` and `BayesianApproach`. `DeeplearningApproach`
supplies a deep-learning based prediction tool for kcat prediction,
while `BayesianApproach` supplies an automatic Bayesian based pipeline
to construct ecModels using the predicted kcats.

Usage
-----

-   Please check the instruction `README` file under these two section
    `Bayesianapproach` and `DeeplearningApproach` for reporducing all figures in
    the paper.
-   For people who are interested in using the trained deep-learning
    model for their own kcat prediction, we supplied an example. please
    check usage for **detailed information** in the file
    [DeeplearningApproach/README](https://github.com/SysBioChalmers/DLKcat/tree/master/DeeplearningApproach)
    under the `DeeplearningApproach`.

    > -   `input` for the prediction is the `Protein sequence` and
    >     `Substrate SMILES structure/Substrate name`, please check the
    >     file in
    >     [DeeplearningApproach/Code/example/input.tsv](https://github.com/SysBioChalmers/DLKcat/tree/master/DeeplearningApproach/Code/example)
    > -   `output` is the correponding `kcat` value

Citation
-----

- Please cite the paper [Deep learning-based kcat prediction enables improved enzyme-constrained model reconstruction](https://www.nature.com/articles/s41929-022-00798-z)""


Notes
-------
We noticed there is a mismatch of reference list in Supplementary Table 2 of the publication, therefore we made an update for that. New supplementary Tables can be found [here](https://github.com/SysBioChalmers/DLKcat/tree/master/DeeplearningApproach/Results/figures)

Contact
-------

-   Feiran Li ([@feiranl](https://github.com/feiranl)), Chalmers
    University of Technology, Gothenburg, Sweden
-   Le Yuan ([@le-yuan](https://github.com/le-yuan)), Chalmers
    University of Technology, Gothenburg, Sweden

Last update: 2022-04-09
