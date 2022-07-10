""" run_knn.py

This file conducts a KNN baseline of the model

"""

import json
import math
import argparse
import itertools
import multiprocessing as mp
from typing import List
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import r2_score, mean_squared_error

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

class KMERFeaturizer:
    """KMERFeaturizer."""
    def __init__(
        self,
        ngram_min: int = 2,
        ngram_max: int = 4,
        unnormalized: bool = False,
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

        self.vectorizer = CountVectorizer(ngram_range=(self.ngram_min,
                                                       self.ngram_max),
                                          analyzer="char")

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


class MorganFeaturizer:
    """MorganFeaturizer."""
    def __init__(self):
        pass

    def _mol_to_fp(self, mol: Chem.Mol) -> np.ndarray:
        """Convert mol to fingerprint"""
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        array = np.zeros((0, ), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        return array

    def featurize(self, mol_list: List[str]) -> List[np.ndarray]:
        """featurize."""
        ret = []
        for mol in tqdm(mol_list):
            mol = Chem.MolFromSmiles(mol)
            ret.append(self._mol_to_fp(mol))
        return ret


class KNNModel:
    """KNNModel"""
    def __init__(self, n=3, seq_dist_weight=5, comp_dist_weight=1, **kwargs):
        self.n = n
        self.seq_dist_weight = seq_dist_weight
        self.comp_dist_weight = comp_dist_weight

    def cosine_dist(self, train_objs, test_objs):
        """compute cosine_dist"""
        numerator = train_objs[:, None, :] * test_objs[None, :, :]
        numerator = numerator.sum(-1)

        norm = lambda x: (x**2).sum(-1)**(0.5)

        denominator = norm(train_objs)[:, None] * norm(test_objs)[None, :]
        denominator[denominator == 0] = 1e-12
        cos_dist = 1 - numerator / denominator

        return cos_dist

    def fit(self, train_seqs, train_comps, train_vals, val_seqs, val_comps,
            val_vals) -> None:
        self.train_seqs = train_seqs
        self.train_comps = train_comps
        self.train_vals = train_vals

    def predict(self, test_seqs, test_comps) -> np.ndarray:
        # Compute test dists
        test_seq_dists = self.cosine_dist(self.train_seqs, test_seqs)
        test_comp_dists = self.cosine_dist(self.train_comps, test_comps)
        total_dists = (self.seq_dist_weight * test_seq_dists +
                       self.comp_dist_weight * test_comp_dists)

        smallest_dists = np.argsort(total_dists, 0)

        top_n = smallest_dists[:self.n, :]
        ref_vals = self.train_vals[top_n]
        mean_preds = np.mean(ref_vals, 0)
        return mean_preds


def shuffle_dataset(dataset, seed):
    """shuffle_dataset."""
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    """split_dataset."""
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


def single_trial(model_params, train_data, dev_data, test_data):
    """Conduct a single trial with given model params"""

    # Unpack data
    train_seq_feats, train_sub_feats, train_vals = train_data
    dev_seq_feats, dev_sub_feats, dev_vals = dev_data
    test_seq_feats, test_sub_feats, test_vals = test_data

    # Create model
    knn_model = KNNModel(**model_params)

    knn_model.fit(
        train_seq_feats,
        train_sub_feats,
        train_vals,
        dev_seq_feats,
        dev_sub_feats,
        dev_vals,
    )

    # Conduct analysis on val and test set
    outputs = {}
    outputs.update(model_params)
    for dataset, seq_feats, sub_feats, targs in zip(
        ["val", "test"],
        [dev_seq_feats, test_seq_feats],
        [dev_sub_feats, test_sub_feats],
        [dev_vals, test_vals],
    ):

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
        RMSE = np.sqrt((SAE**2).mean())

        results = {
            f"{dataset}_mae": MAE,
            f"{dataset}_RMSE": RMSE,
            f"{dataset}_r2": r2
        }

        outputs.update(results)
    return outputs



def make_scatter(df, title="scatter_all.pdf"): 
    """make_scatter"""
    experimental_values, predicted_values = df['kcat'].values, df['pred'].values
    correlation, p_value = stats.pearsonr(experimental_values, predicted_values)

    r2 = r2_score(experimental_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(experimental_values, predicted_values))

    allData = pd.DataFrame(list(zip(experimental_values, predicted_values)))
    allData.columns = ['Experimental value', 'Predicted value']

    plt.figure(figsize=(1.5,1.5))

    # To solve the 'Helvetica' font cannot be used in PDF file
    # https://stackoverflow.com/questions/59845568/the-pdf-backend-does-not-currently-support-the-selected-font
    # rc('text', usetex=True) 
    rc('font',**{'family':'serif','serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    # plt.rc('text', usetex=True)

    plt.axes([0.12,0.12,0.83,0.83])

    plt.tick_params(direction='in')
    plt.tick_params(which='major',length=1.5)
    plt.tick_params(which='major',width=0.4)

    kcat_values_vstack = np.vstack([experimental_values, predicted_values])
    experimental_predicted = stats.gaussian_kde(kcat_values_vstack)(kcat_values_vstack)

    ax = plt.scatter(x=experimental_values, y=predicted_values, 
                     c=experimental_predicted, s=3, edgecolor=[])

    cbar = plt.colorbar(ax)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label('Density', size=7)

    plt.text(-4.7, 6.9, f'r = {correlation:.2f}', fontweight ="normal", fontsize=6)
    plt.text(-4.7, 5.9, f'P value = {p_value:.2E}', fontweight ="normal", fontsize=6)
    plt.text(-4.7, 4.8, f'N = {len(experimental_values)}', fontweight ="normal", fontsize=6)

    plt.rcParams['font.family'] = 'Helvetica'

    plt.xlabel("Experimental $k$$_\mathregular{cat}$ value", fontdict={'weight': 'normal', 'fontname': 'Helvetica', 'size': 7}, fontsize=7)
    plt.ylabel('Predicted $k$$_\mathregular{cat}$ value',fontdict={'weight': 'normal', 'fontname': 'Helvetica', 'size': 7},fontsize=7)

    plt.xticks([-6, -4, -2, 0, 2, 4, 6, 8])
    plt.yticks([-6, -4, -2, 0, 2, 4, 6, 8])

    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)

    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)

    plt.savefig(title, dpi=400, bbox_inches='tight')
    plt.close()

def get_args():
    """get_args."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--hyperopt", default=False, action="store_true")
    parser.add_argument("--run-model", default=False, action="store_true")
    parser.add_argument("--workers", default=1, type=int)
    args = parser.parse_args()
    return args

def main():
    # args
    args = get_args()
    debug = args.debug
    hyperopt = args.hyperopt
    workers = args.workers
    run_model= args.run_model

    # Parse preprocessed data
    dir_input = Path("../../Data/database/Kcat_combination_0918.json")
    with open(dir_input, "r") as fp:
        json_obj = json.load(fp)

    if debug:
        json_obj = json_obj[:500]

    # Parse data and split exactly as in DLKCat
    # Need to extract morgan FP's and kmer feature vector
    i = 0
    seqs, subs, vals, ecs = [], [], [], []
    for data in tqdm(json_obj):
        smiles = data["Smiles"]
        sequence = data["Sequence"]
        # print(smiles)
        Kcat = data["Value"]
        ec = data['ECNumber']
        if "." not in smiles and float(Kcat) > 0:
            seqs.append(sequence)
            subs.append(smiles)
            vals.append(math.log2(float(Kcat)))
            ecs.append(ec)
    seqs, subs, vals, ecs = np.array(seqs), np.array(subs), np.array(vals), np.array(ecs)

    # Shuffle for split (as in DLKCat)
    dataset = np.arange(len(seqs))
    dataset = shuffle_dataset(dataset, 1234)
    dataset_train, dataset_ = split_dataset(dataset, 0.8)
    dataset_dev, dataset_test = split_dataset(dataset_, 0.5)

    # Split data itself
    train_seqs, train_subs, train_vals, dev_ecs = (
        seqs[dataset_train],
        subs[dataset_train],
        vals[dataset_train],
        ecs[dataset_train]
    )
    dev_seqs, dev_subs, dev_vals, dev_ecs = (
        seqs[dataset_dev],
        subs[dataset_dev],
        vals[dataset_dev],
        ecs[dataset_dev]
    )
    test_seqs, test_subs, test_vals, test_ecs = (
        seqs[dataset_test],
        subs[dataset_test],
        vals[dataset_test],
        ecs[dataset_test]
    )

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
    knn_params = {"n": 3, "seq_dist_weight": 5, "comp_dist_weight": 1}

    if hyperopt:
        train_data = (train_seq_feats, train_sub_feats, train_vals)
        dev_data = (dev_seq_feats, dev_sub_feats, dev_vals)
        test_data = (test_seq_feats, test_sub_feats, test_vals)

        # Make parameter grid
        model_params = {
            "n": [2, 3, 4, 5],
            "seq_dist_weight": [1, 5, 10],
            "comp_dist_weight": [1, 3, 5],
        }

        key, values = zip(*model_params.items())
        combos = [
            dict(zip(key, val_combo))
            for val_combo in itertools.product(*values)
        ]
        trial_fn = partial(single_trial,
                           train_data=train_data,
                           dev_data=dev_data,
                           test_data=test_data)

        print("Starting grid sweep")

        # Parallel
        with mp.Pool(workers) as p:
            res_list = list(tqdm(p.imap(trial_fn, combos), total=len(combos)))

        res = json.dumps(res_list, indent=2)
        with open("knn_hyperopt_out.json", "w") as fp:
            fp.write(res)
        # Set best params
        knn_params = sorted(res_list, key=lambda x: x['val_mae'])[0]
        print(f"Setting model params to best hyperopt:\n{json.dumps(knn_params,indent=2)}")

    if run_model:
        print("Running model")

        knn_model = KNNModel(**knn_params)
        knn_model.fit(
            train_seq_feats,
            train_sub_feats,
            train_vals,
            dev_seq_feats,
            dev_sub_feats,
            dev_vals,
        )
        inds = np.arange(len(test_seq_feats))
        num_splits = min(100, len(inds))
        ars = np.array_split(inds, num_splits)
        ar_vec = []
        for ar in tqdm(ars):
            test_preds = knn_model.predict(test_seq_feats[ar],
                                           test_sub_feats[ar])
            ar_vec.append(test_preds)
        test_preds = np.concatenate(ar_vec)

        # Evaluation
        print("Conducting evaluation")
        true_vals_corrected = np.log10(np.power(2, test_vals))
        predicted_vals_corrected = np.log10(np.power(2, test_preds))
        SAE = np.abs(predicted_vals_corrected - true_vals_corrected)
        MAE = np.mean(SAE)
        RMSE = np.sqrt((SAE**2).mean())
        r2 = r2_score(test_vals, test_preds)
        correlation, p_value = stats.pearsonr(test_vals, test_preds)

        # Dump outputs to file
        outputs = { "MAE": MAE, "RMSE": RMSE, "R2": r2, "R": correlation}
        print(json.dumps(outputs, indent=2))

        # Get all entries where test is in train where test is in train 
        train_subs_set, train_seqs_set = set(train_subs), set(train_seqs)
        dev_subs_set, dev_seqs_set = set(dev_subs), set(dev_seqs)
        sub_in_train = np.array([i in train_subs_set or 
                                 i in dev_subs_set
                                 for i in test_subs])
        seq_in_train = np.array([i in train_seqs_set or 
                                 i in dev_seqs_set
                                 for i in test_seqs])
        output_data = list(
            zip(test_seqs, test_subs, true_vals_corrected, 
                predicted_vals_corrected, test_ecs, sub_in_train, 
                seq_in_train)
        )
        index = ["seqs", "subs", "kcat", "pred", "ec", 
                 "sub_in_train", "seq_in_train"]
        df = pd.DataFrame(output_data, columns=index)

        # Make ec sub levels
        for ec_level in [1,2,3,4]:
            ec_sub_num = [i.rsplit(".", 4 - ec_level)[0] for i in df['ec'].values]
            df[f'ec_{ec_level}'] = ec_sub_num
        df.to_csv("knn_test_preds.tsv", sep="\t")

    # Make scatter
    df = pd.read_csv("knn_test_preds.tsv", sep="\t")

    make_scatter(df, title="test_scatter.pdf")
    # Create subset where either seq or sub not in train
    df_subset = df[np.logical_or(~df['seq_in_train'].values,
                                 ~df['sub_in_train'].values,)
                   ]
    make_scatter(df_subset, title="test_scatter_subset.pdf")
    ## 

if __name__ == "__main__":
    main()
