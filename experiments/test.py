import numpy as np
from ampligraph.datasets import load_wn18, load_fb15k_237
from ampligraph.latent_features import ComplEx
from ampligraph.evaluation import evaluate_performance, mrr_score, hits_at_n_score
import os
import sys
def main():
    # load Wordnet18 dataset:
    # X = load_wn18()
    X = load_fb15k_237()
    modify_flag = False

    # Initialize a ComplEx neural embedding model with pairwise loss function:
    # The model will be trained for 300 epochs.
    model = ComplEx(batches_count=10, seed=0, epochs=30, k=150, eta=10,
                    # Use adam optimizer with learning rate 1e-3
                    optimizer='adam', optimizer_params={'lr': 1e-3},
                    # Use pairwise loss with margin 0.5
                    loss='pairwise', loss_params={'margin': 0.5},
                    # Use L2 regularizer with regularizer weight 1e-5
                    regularizer='LP', regularizer_params={'p': 2, 'lambda': 1e-5},
                    # Enable stdout messages (set to false if you don't want to display)
                    verbose=True) #, modify_flag = modify_flag)

    if False:
        # ground truth params (have not tried yet)
        # k: 350; epochs: 4000; eta: 30; loss: self_adversarial; loss_params: alpha: 1; margin: 0.5; optimizer: adam; optimizer_params: lr: 0.0001; seed: 0; batches_count: 50
        model = ComplEx(batches_count=50, seed=0, epochs=4000, k=350, eta=30,
                        # Use adam optimizer with learning rate 1e-3
                        optimizer='adam', optimizer_params={'lr': 1e-4},
                        # Use pairwise loss with margin 0.5
                        loss='self_adversarial', loss_params={'margin': 0.5, 'alpha': 1},
                        # Use L2 regularizer with regularizer weight 1e-5
                        regularizer='LP', regularizer_params={'p': 2, 'lambda': 1e-5},
                        # Enable stdout messages (set to false if you don't want to display)
                        verbose=True, modify_flag = modify_flag)

    # For evaluation, we can use a filter which would be used to filter out
    # positives statements created by the corruption procedure.
    # Here we define the filter set by concatenating all the positives
    filter = np.concatenate((X['train'], X['valid'], X['test']))

    # Fit the model on training and validation set
    model.fit(X['train'],
              early_stopping=True,
              early_stopping_params= \
                  {
                      'x_valid': X['valid'],  # validation set
                      'criteria': 'hits10',  # Uses hits10 criteria for early stopping
                      'burn_in': 100,  # early stopping kicks in after 100 epochs
                      'check_interval': 20,  # validates every 20th epoch
                      'stop_interval': 5,  # stops if 5 successive validation checks are bad.
                      'x_filter': filter,  # Use filter for filtering out positives
                      'corruption_entities': 'all',  # corrupt using all entities
                      'corrupt_side': 's+o'  # corrupt subject and object (but not at once)
                  }
              )

    # Run the evaluation procedure on the test set (with filtering).
    # To disable filtering: filter_triples=None
    # Usually, we corrupt subject and object sides separately and compute ranks

    # restore model
    sys.exit()


    # import the inspect_checkpoint library

    sys.exit()
    ranks = evaluate_performance(X['test'],
                                 model=model,
                                 filter_triples=filter,
                                 use_default_protocol=True,  # corrupt subj and obj separately while evaluating
                                 verbose=True)

    # compute and print metrics:
    mrr = mrr_score(ranks)
    hits_10 = hits_at_n_score(ranks, n=10)
    print("MRR: %f, Hits@10: %f" % (mrr, hits_10))


if __name__ == "__main__":
    main()