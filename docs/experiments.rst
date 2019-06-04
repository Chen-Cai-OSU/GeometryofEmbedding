Performance
===========


Predictive Performance
----------------------

We report the filtered MR, MRR, Hits@1,3,10 for the most common datasets used in literature.


FB15K-237 
---------

========== ======== ====== ======== ======== ========== ========================
  Model       MR     MRR    Hits@1   Hits@3   Hits\@10      Hyperparameters
========== ======== ====== ======== ======== ========== ========================
  TransE    153     0.31    0.22     0.35      0.51      k: 1000;
                                                         epochs: 4000;
                                                         eta: 50;
                                                         loss: self_adversarial;
                                                         loss_params:
                                                         alpha: 0.5;
                                                         margin: 5;
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0001;
                                                         embedding_model_params:
                                                         norm: 1;
                                                         normalize_ent_emb: false;
                                                         seed: 0;
                                                         batches_count: 60

 DistMult   568     0.29      0.20     0.32      0.47    k: 400;
                                                         epochs: 4000;
                                                         eta: 50;
                                                         loss: self_adversarial;
                                                         loss_params:
                                                         alpha: 1;
                                                         margin: 1;
                                                         regularizer: LP;
                                                         regularizer_params:
                                                         lambda: 0.0001;
                                                         p: 2;
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0005;
                                                         seed: 0;
                                                         normalize_ent_emb: false;
                                                         batches_count: 50

   ComplEx  519     0.30      0.20     0.33      0.48    k: 350;
                                                         epochs: 4000;
                                                         eta: 30;
                                                         loss: self_adversarial;
                                                         loss_params:
                                                         alpha: 1;
                                                         margin: 0.5;
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0001;
                                                         seed: 0;
                                                         batches_count: 50
                                                         

   HolE     297     0.28       0.19     0.31       0.46  k: 350;
                                                         epochs: 4000;
                                                         eta: 30;
                                                         loss: self_adversarial;
                                                         loss_params:
                                                         alpha: 1
                                                         margin: 0.5;
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0001;
                                                         seed: 0;
                                                         batches_count: 50
                                                         

========== ======== ====== ======== ======== ========== ========================

.. note:: FB15K-237 validation and test sets include triples with entities that do not occur 
    in the training set. We found 8 unseen entities in the validation set and 29 in the test set.
    In the experiments we excluded the triples where such entities appear (9 triples in from the validation
    set and 28 from the test set).



WN18RR 
------

========== ========= ====== ======== ======== ========== =======================
  Model       MR      MRR    Hits@1   Hits@3   Hits\@10      Hyperparameters
========== ========= ====== ======== ======== ========== =======================
  TransE    1536      0.23    0.07     0.35      0.51     k: 200;
                                                          epochs: 4000;
                                                          eta: 20;
                                                          loss: self_adversarial;
                                                          loss_params:
                                                          margin: 1;
                                                          optimizer: adam;
                                                          optimizer_params:
                                                          lr: 0.0001;
                                                          regularizer: LP;
                                                          regularizer_params:
                                                          lambda: 1.0e-05;
                                                          p: 1;
                                                          seed: 0;
                                                          normalize_ent_emb: false;
                                                          embedding_model_params:
                                                          norm: 1;
                                                          batches_count: 100;

 DistMult   6853      0.44    0.42     0.45      0.50     k: 200;
                                                          epochs: 4000;
                                                          eta: 20;
                                                          loss: self_adversarial;
                                                          loss_params:
                                                          margin: 1;
                                                          optimizer: adam;
                                                          optimizer_params:
                                                          lr: 0.0005;
                                                          seed: 0;
                                                          normalize_ent_emb: false;
                                                          batches_count: 25

 ComplEx    8214      0.44    0.41     0.45      0.50     k: 200;
                                                          epochs: 4000;
                                                          eta: 20;
                                                          loss: nll;
                                                          loss_params:
                                                          margin: 1;
                                                          optimizer: adam;
                                                          optimizer_params:
                                                          lr: 0.0005;
                                                          seed: 0;
                                                          batches_count: 10;
                                                          
   HolE     7305      0.47    0.43     0.48      0.53     k: 200;
                                                          epochs: 4000;
                                                          eta: 20;
                                                          loss: self_adversarial;
                                                          loss_params:
                                                          margin: 1;
                                                          optimizer: adam;
                                                          optimizer_params:
                                                          lr: 0.0005;
                                                          seed: 0;
                                                          batches_count: 50

========== ========= ====== ======== ======== ========== =======================

.. note:: WN18RR validation and test sets include triples with entities that do not occur
    in the training set. We found 198 unseen entities in the validation set and 209 in the test set.
    In the experiments we excluded the triples where such entities appear (210 triples in from the validation
    set and 210 from the test set).


YAGO3-10
--------

======== ======== ====== ======== ======== ========= =========================
 Model      MR     MRR    Hits@1   Hits@3   Hits\@10      Hyperparameters
======== ======== ====== ======== ======== ========= =========================
TransE   574      0.24   0.15     0.26     0.41       k: 1000;
                                                      epochs: 4000;
                                                      eta: 50;
                                                      loss: self_adversarial;
                                                      loss_params:
                                                      alpha: 0.5;
                                                      margin: 5;
                                                      optimizer: adam;
                                                      optimizer_params:
                                                      lr: 0.0001;
                                                      embedding_model_params:
                                                      norm: 1;
                                                      normalize_ent_emb: false;
                                                      seed: 0;
                                                      batches_count: 150
                                                      
DistMult 4903     0.49   0.41     0.54     0.63       k: 400;
                                                      epochs: 4000;
                                                      eta: 50;
                                                      loss: self_adversarial;
                                                      loss_params:
                                                      alpha: 1;
                                                      margin: 1;
                                                      optimizer: adam;
                                                      optimizer_params:
                                                      lr: 0.0005;
                                                      regularizer: LP;
                                                      regularizer_params:
                                                      lambda: 0.0001;
                                                      p: 2;
                                                      seed: 0;
                                                      normalize_ent_emb: false;
                                                      batches_count: 100

ComplEx  7266     0.50   0.42     0.55     0.65       k: 350;
                                                      epochs: 4000;
                                                      eta: 30;
                                                      loss: self_adversarial;
                                                      loss_params:
                                                      alpha: 1;
                                                      margin: 0.5;
                                                      optimizer: adam;
                                                      optimizer_params:
                                                      lr: 0.0001;
                                                      seed: 0;
                                                      batches_count: 100

HolE     6201     0.50   0.41     0.55     0.65       k: 350;
                                                      epochs: 4000;
                                                      eta: 30;
                                                      loss: self_adversarial;
                                                      loss_params:
                                                      alpha: 1;
                                                      margin: 0.5;
                                                      optimizer: adam;
                                                      optimizer_params:
                                                      lr: 0.0001;
                                                      seed: 0;
                                                      batches_count: 100
======== ======== ====== ======== ======== ========= =========================                                                        



.. note:: YAGO3-10 validation and test sets include triples with entities that do not occur
    in the training set. We found 22 unseen entities in the validation set and 18 in the test set.
    In the experiments we excluded the triples where such entities appear (22 triples in from the validation
    set and 18 from the test set).


FB15K
-----


.. warning::
    The dataset includes a large number of inverse relations, and its use in experiments has been deprecated.
    Use FB15k-237 instead.


========== ======== ====== ======== ======== ========== ========================
  Model       MR     MRR    Hits@1   Hits@3   Hits\@10      Hyperparameters
========== ======== ====== ======== ======== ========== ========================
  TransE    105      0.55    0.39     0.68      0.79     k: 150;
                                                         epochs: 4000;
                                                         eta: 5;
                                                         loss: pairwise;
                                                         loss_params:
                                                         margin: 0.5;
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0001;
                                                         regularizer: LP;
                                                         regularizer_params:
                                                         lambda: 0.0001;
                                                         p: 2;
                                                         seed: 0;
                                                         embedding_model_params:
                                                         norm: 1;
                                                         normalize_ent_emb: false;
                                                         batches_count: 10

 DistMult   177      0.79    0.74     0.82      0.86     k: 200;
                                                         epochs: 4000;
                                                         eta: 20;
                                                         loss: self_adversarial;
                                                         loss_params:
                                                         margin: 1;
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0005;
                                                         seed: 0;
                                                         normalize_ent_emb: false;
                                                         batches_count: 50

 ComplEx    188      0.79    0.76     0.82      0.86     k: 200;
                                                         epochs: 4000;
                                                         eta: 20;
                                                         loss: self_adversarial;
                                                         loss_params:
                                                         margin: 1;
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0005;
                                                         seed: 0;
                                                         batches_count: 100

   HolE     212      0.80    0.76     0.83      0.87     k: 200;
                                                         epochs: 4000;
                                                         eta: 20;
                                                         loss: self_adversarial;
                                                         loss_params:
                                                         margin: 1;
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0005;
                                                         seed: 0;
                                                         batches_count: 50

========== ======== ====== ======== ======== ========== ========================

WN18
----

.. warning::
    The dataset includes a large number of inverse relations, and its use in experiments has been deprecated.
    Use WN18RR instead.


========== ======== ====== ======== ======== ========== ========================
  Model       MR     MRR    Hits@1   Hits@3   Hits\@10      Hyperparameters
========== ======== ====== ======== ======== ========== ========================
TransE     446      0.50    0.18     0.81      0.89     k: 150;
                                                        epochs: 4000;
                                                        eta: 5;
                                                        loss: pairwise;
                                                        loss_params:
                                                        margin: 0.5;
                                                        optimizer: adam;
                                                        optimizer_params:
                                                        lr: 0.0001;
                                                        regularizer: LP;
                                                        regularizer_params:
                                                        lambda: 0.0001;
                                                        p: 2;
                                                        embedding_model_params:
                                                        norm: 1;
                                                        normalize_ent_emb: false;
                                                        seed: 0;
                                                        batches_count: 10
                                                        
 DistMult   746      0.83    0.73     0.92      0.95     k: 200;
                                                         epochs: 4000;
                                                         eta: 20;
                                                         loss: nll;
                                                         loss_params:
                                                         margin: 1;
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0005;
                                                         seed: 0;
                                                         normalize_ent_emb: false;
                                                         batches_count: 50

 ComplEx    715      0.94    0.94     0.95      0.95     k: 200;
                                                         epochs: 4000;
                                                         eta: 20;
                                                         loss: nll;
                                                         loss_params:
                                                         margin: 1;
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0005;
                                                         seed: 0;
                                                         batches_count: 50
                                                         
   HolE     658      0.94    0.93     0.94      0.95     k: 200;
                                                         epochs: 4000;
                                                         eta: 20;
                                                         loss: self_adversarial;
                                                         loss_params:
                                                         margin: 1;
                                                         optimizer: adam;
                                                         optimizer_params:
                                                         lr: 0.0005;
                                                         seed: 0;
                                                         batches_count: 50
                                                         
========== ======== ====== ======== ======== ========== ========================

To reproduce the above results: ::
    
    $ cd experiments
    $ python predictive_performance.py


.. note:: Running ``predictive_performance.py`` on all datasets, for all models takes ~43 hours on
    an Intel Xeon Gold 6142, 64 GB Ubuntu 16.04 box equipped with a Tesla V100 16GB.



Experiments can be limited to specific models-dataset combinations as follows: ::

    $ python predictive_performance.py -h
    usage: predictive_performance.py [-h] [-d {fb15k,fb15k-237,wn18,wn18rr,yago310}]
                                     [-m {complex,transe,distmult,hole}]

    optional arguments:
      -h, --help            show this help message and exit
      -d {fb15k,fb15k-237,wn18,wn18rr,yago310}, --dataset {fb15k,fb15k-237,wn18,wn18rr,yago310}
      -m {complex,transe,distmult,hole}, --model {complex,transe,distmult,hole}

Runtime Performance
-------------------

Training the models on FB15K-237 (``k=200, eta=2, batches_count=100, loss=nll``), on an Intel Xeon Gold 6142, 64 GB
Ubuntu 16.04 box equipped with a Tesla V100 16GB gives the following runtime report:

======== ==============
model     seconds/epoch
======== ==============
ComplEx     3.19
TransE      3.26
DistMult    2.61
HolE        3.21
======== ==============
