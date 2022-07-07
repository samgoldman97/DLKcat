""" run_knn.py

This file conducts a KNN baseline of the model

"""

import copy
import argparse
import pickle
import numpy as np
from functools import partial
import itertools
from typing import List
from pathlib import Path
import json
from tqdm import tqdm
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import r2_score
import math
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import multiprocessing as mp

class KMERFeaturizer(object):
    """KMERFeaturizer.
    """

    def __init__(
        self,
        ngram_min: int = 2,
        ngram_max: int = 4,
        unnormalized: bool = False,
        **kwargs,
    ):
        """__init__.

        Args:
            ngram_min (int): ngram_min
            ngram_max (int): ngram_max
            unnormalized (bool): normalize
            kwargs: kwargs
        """

        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.normalize = not (unnormalized)

        self.vectorizer = CountVectorizer(
            ngram_range=(self.ngram_min, self.ngram_max), analyzer="char"
        )

        # If false, fit, otherwise just run
        self.is_fit = False

    def featurize(self, seq_list: List[str]) -> List[np.ndarray]:
        """featurize.

        On first settles on fixed kmer components

        Args:
            seq_list (List[str]): seq_list containing strings of smiles

        Returns:
            np.ndarray: of features
        """
        if not self.is_fit:
            self.vectorizer.fit(seq_list)
            self.is_fit = True
        output = self.vectorizer.transform(seq_list)
        output = np.asarray(output.todense())

        # If this is true, normalize the sequence
        if self.normalize:
            output = output / output.sum(1).reshape(-1, 1)

        return list(output)

class MorganFeaturizer(object):
    """MorganFeaturizer.
    """

    def __init__(
        self,
        **kwargs,
    ):
        pass

    def _mol_to_fp(self, mol: Chem.Mol) -> np.ndarray:
        """Convert mol to fingerprint"""
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        array = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        return array

    def featurize(self, mol_list: List[str]) -> List[np.ndarray]:
        """ featurize. """
        ret = []
        for mol in tqdm(mol_list):
            mol = Chem.MolFromSmiles(mol)
            ret.append(self._mol_to_fp(mol))
        return ret

class KNNModel(object): 
    """ KNNModel """

    def __init__(self, n=3, seq_dist_weight=5, comp_dist_weight=1):
        self.n = n
        self.seq_dist_weight = seq_dist_weight
        self.comp_dist_weight = comp_dist_weight

    def cosine_dist(self, train_objs, test_objs): 
        """ compute cosine_dist """
        numerator =  train_objs[:, None, :] * test_objs[None, : , :]
        numerator =  numerator.sum(-1)

        norm = lambda x: (x ** 2).sum(-1) ** (0.5)

        denominator =  norm(train_objs)[:, None] * norm(test_objs)[None, :]
        denominator[denominator == 0] = 1e-12
        cos_dist = 1 - numerator / denominator

        return cos_dist

    def fit(self, train_seqs, train_comps, train_vals, 
            val_seqs, val_comps, val_vals) -> None: 
        self.train_seqs = train_seqs
        self.train_comps = train_comps
        self.train_vals = train_vals

        # TODO: Actually fit the seq_dist_weight and sub_dist_weight in this
        # method

    def predict(self, test_seqs, test_comps) -> np.ndarray:
        # Compute test dists
        test_seq_dists = self.cosine_dist(self.train_seqs, test_seqs)
        test_comp_dists = self.cosine_dist(self.train_comps, test_comps)
        total_dists = self.seq_dist_weight * test_seq_dists + self.comp_dist_weight * test_comp_dists

        smallest_dists = np.argsort(total_dists, 0)

        top_n = smallest_dists[:self.n, :]
        ref_vals = self.train_vals[top_n]
        mean_preds = np.mean(ref_vals, 0)
        return mean_preds

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


def single_trial(model_params, train_data, dev_data, test_data):
    """ Conduct a single trial with given model params """

    # Unpack data
    train_seq_feats, train_sub_feats, train_vals = train_data
    dev_seq_feats, dev_sub_feats, dev_vals = dev_data
    test_seq_feats, test_sub_feats, test_vals = test_data

    # Create model
    knn_model = KNNModel(**model_params)

    knn_model.fit(train_seq_feats, train_sub_feats, train_vals,
                  dev_seq_feats, dev_sub_feats, dev_vals)


    # Conduct analysis on val and test set
    outputs = {}
    outputs.update(model_params)
    for dataset, seq_feats, sub_feats, targs in zip(["val", "test"],
                                                    [dev_seq_feats, test_seq_feats],
                                                    [dev_sub_feats, test_sub_feats],
                                                    [dev_vals, test_vals]):

        inds = np.arange(len(seq_feats))
        num_splits = min(50, len(inds))
        ars = np.array_split(inds, num_splits)
        ar_vec = []
        for ar in ars:
            test_preds = knn_model.predict(seq_feats[ar], sub_feats[ar])
            ar_vec.append(test_preds)

        test_preds = np.concatenate(ar_vec)

        # Evaluation
        true_vals_corrected = np.log10(np.power(2, targs))
        predicted_vals_corrected = np.log10(np.power(2, test_preds))
        SAE = np.abs(predicted_vals_corrected - true_vals_corrected)
        MAE = np.mean(SAE)
        r2 = r2_score(predicted_vals_corrected, true_vals_corrected)
        RMSE = np.sqrt((SAE ** 2).mean())

        results = {
            f"{dataset}_mae": MAE,
            f"{dataset}_RMSE": RMSE,
            f"{dataset}_r2": r2
        }

        outputs.update(results)
    return outputs



if __name__ == "__main__":
    """Load data."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--hyperopt", default=False, action="store_true")
    args = parser.parse_args()
    debug = args.debug
    hyperopt = args.hyperopt

    # Parse preprocessed data
    dir_input = Path('../../Data/database/Kcat_combination_0918.json')
    with open(dir_input, "r") as fp:
        json_obj = json.load(fp)

    if debug:
        json_obj = json_obj[:500]

    # Parse it out; code taken  from preprocess_all.py
    # Need to extract morgan FP's and kmer feature vector
    i = 0
    seqs, subs, vals = [], [], []
    for data in tqdm(json_obj):
        smiles = data['Smiles']
        sequence = data['Sequence']
        # print(smiles)
        Kcat = data['Value']
        if "." not in smiles and float(Kcat) > 0:
            seqs.append(sequence)
            subs.append(smiles)
            vals.append(math.log2(float(Kcat)))

    seqs, subs, vals = np.array(seqs), np.array(subs), np.array(vals)

    # Shuffle, but shuffle the _indices_
    dataset = np.arange(len(seqs))
    dataset = shuffle_dataset(dataset, 1234)
    dataset_train, dataset_ = split_dataset(dataset, 0.8)
    dataset_dev, dataset_test = split_dataset(dataset_, 0.5)

    # Split data itself
    train_seqs, train_subs, train_vals = seqs[dataset_train], subs[dataset_train], vals[dataset_train]
    dev_seqs, dev_subs, dev_vals = seqs[dataset_dev], subs[dataset_dev], vals[dataset_dev]
    test_seqs, test_subs, test_vals = seqs[dataset_test], subs[dataset_test], vals[dataset_test]

    # Featurize compounds with morgan fingerprints
    sub_featurizer = MorganFeaturizer()
    train_sub_feats = np.vstack(sub_featurizer.featurize(train_subs))
    dev_sub_feats = np.vstack(sub_featurizer.featurize(dev_subs))
    test_sub_feats = np.vstack(sub_featurizer.featurize(test_subs))

    # Featurizer sequences with kmers
    print("Featurizing sequences")
    seq_featurizer = KMERFeaturizer(ngram_min=2,
                                    ngram_max=3,
                                    unnormalized=True)
    train_seq_feats = np.vstack(seq_featurizer.featurize(train_seqs))
    dev_seq_feats = np.vstack(seq_featurizer.featurize(dev_seqs))
    test_seq_feats = np.vstack(seq_featurizer.featurize(test_seqs))

    if hyperopt:
        train_data = (train_seq_feats, train_sub_feats, train_vals)
        dev_data = (dev_seq_feats, dev_sub_feats, dev_vals)
        test_data = (test_seq_feats, test_sub_feats, test_vals)


        # Make parameter grid
        model_params = {"n": [1, 3, 5, 10],
                        "seq_dist_weight": [1, 5, 10],
                        "comp_dist_weight": [1, 5, 10]
                        }

        key, values = zip(*model_params.items())
        combos = [dict(zip(key, val_combo))
                  for val_combo in itertools.product(*values)
                  ]
        trial_fn = partial(single_trial, train_data=train_data, 
                           dev_data=dev_data, test_data=test_data)

        print("Starting grid sweep")

        # Not parallel
        #res_list = []
        #for model_param in combos:
        #    out_results = trial_fn(model_param)
        #    res_list.append(out_results)

        # Parallel
        with mp.Pool(16) as p:
            res_list = list(tqdm(p.imap(trial_fn, combos),
                                 total=len(combos)))

        res = json.dumps(res_list, indent=2)
        print(res)
        with open("out.json", "w") as fp:
            fp.write(res)

    else:
        print("Running model")

        knn_model = KNNModel()
        knn_model.fit(train_seq_feats, train_sub_feats, train_vals, 
                      dev_seq_feats, dev_sub_feats, dev_vals)
        inds = np.arange(len(test_seq_feats))
        num_splits = min(50, len(inds))
        ars = np.array_split(inds, num_splits)
        ar_vec = []
        for ar in tqdm(ars):
            test_preds = knn_model.predict(test_seq_feats[ar], test_sub_feats[ar])
            ar_vec.append(test_preds)
        test_preds = np.concatenate(ar_vec)

        # Evaluation
        print("Conducting evaluation")
        true_vals_corrected = np.log10(np.power(2, test_vals))
        predicted_vals_corrected = np.log10(np.power(2, test_preds))
        SAE = np.abs(predicted_vals_corrected - true_vals_corrected)
        MAE = np.mean(SAE) 
        RMSE = np.sqrt((SAE ** 2).mean())
        r2 = r2_score(test_vals, test_preds)
        print(f"MAE: {MAE}")
        print(f"RMSE: {RMSE}")
        print(f"R2: {r2}")
