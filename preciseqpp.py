import pyterrier as pt
import pandas as pd
import os
import csv
import pandas as pd
import pyterrier as pt
from pyterrier.measures import *
from scipy.stats import kendalltau
import numpy as np
from collections import defaultdict

from scipy.stats import kendalltau
from collections import defaultdict

def compute_global_kendall_multiqpp(qpp_estimates, true_ap_scores):
    """
    Computes global Kendall's tau across all (query, model) pairs, for each QPP model.

    Parameters:
    - qpp_estimates: dict of {qid: {model: [qpp1, qpp2, ...]}}
    - true_ap_scores: dict of {qid: {model: ap_score}}

    Returns:
    - dict of {qpp_model_index: global_kendall_tau}
    """
    # Accumulate global lists of (QPP estimate, AP) for each QPP model
    qpp_data = defaultdict(lambda: ([], []))  # index: (qpp_list, ap_list)

    for qid in qpp_estimates:
        if qid not in true_ap_scores:
            continue
        for model in qpp_estimates[qid]:
            if model not in true_ap_scores[qid]:
                continue

            preds = qpp_estimates[qid][model]
            ap = true_ap_scores[qid][model]

            for i, pred in enumerate(preds):
                qpp_data[i][0].append(pred)
                qpp_data[i][1].append(ap)

    # Compute Kendall's tau for each QPP model
    global_tau = {}
    for i in sorted(qpp_data.keys()):
        qpp_vals, ap_vals = qpp_data[i]
        if len(qpp_vals) >= 2:
            tau, _ = kendalltau(qpp_vals, ap_vals)
            global_tau[i] = tau

    return global_tau

def compute_modelwise_kendall_multiqpp(qpp_estimates, true_ap_scores):
    """
    Computes Kendall's tau between QPP estimates and AP values across queries, 
    for each QPP model and IR model.

    Parameters:
    - qpp_estimates: dict of {qid: {model: [qpp1, qpp2, ...]}}
    - true_ap_scores: dict of {qid: {model: ap_score}}

    Returns:
    - dict of {model: {qpp_model_index: tau_value}}
    """
    modelwise_tau = {}

    # Get all IR models (from union of inner keys)
    all_models = set()
    for qid in qpp_estimates:
        all_models.update(qpp_estimates[qid].keys())
    for qid in true_ap_scores:
        all_models.update(true_ap_scores[qid].keys())

    for model in sorted(all_models):
        qpp_vals_by_index = defaultdict(list)
        ap_vals = []

        # For each query, collect the QPP predictions and AP for this model
        for qid in qpp_estimates:
            if model not in qpp_estimates[qid] or model not in true_ap_scores.get(qid, {}):
                continue

            qpp_preds = qpp_estimates[qid][model]
            ap_val = true_ap_scores[qid][model]

            for i, pred in enumerate(qpp_preds):
                qpp_vals_by_index[i].append((qid, pred, ap_val))

        modelwise_tau[model] = {}
        for i, vals in qpp_vals_by_index.items():
            qpp_list = [v[1] for v in vals]
            ap_list = [v[2] for v in vals]

            if len(qpp_list) >= 2:
                tau, _ = kendalltau(qpp_list, ap_list)
                modelwise_tau[model][i] = tau

    return modelwise_tau

def compute_perquery_kendall_multiqpp(qpp_estimates, true_ap_scores):
    """
    Computes Kendall's tau correlation between each QPP model and true AP scores, per query.

    Parameters:
    - qpp_estimates: dict of {qid: {model: [qpp1, qpp2, ...]}}
    - true_ap_scores: dict of {qid: {model: ap_score}}

    Returns:
    - dict of {qid: {qpp_model_index: tau_value}}
    """
    perquery_tau = {}

    for qid in qpp_estimates:
        if qid not in true_ap_scores:
            continue

        models_in_common = set(qpp_estimates[qid]) & set(true_ap_scores[qid])
        if len(models_in_common) < 2:
            continue  # Not enough models to compute Kendall's tau

        # For each QPP model index, collect predictions and corresponding AP scores
        # We'll assume all QPP vectors are of the same length
        qpp_len = len(next(iter(qpp_estimates[qid].values())))
        perquery_tau[qid] = {}

        for i in range(qpp_len):
            qpp_vals = []
            ap_vals = []

            for model in sorted(models_in_common):
                try:
                    qpp_val = qpp_estimates[qid][model][i]
                    ap_val = true_ap_scores[qid][model]
                    qpp_vals.append(qpp_val)
                    ap_vals.append(ap_val)
                except (IndexError, KeyError):
                    continue  # Skip any inconsistent entries

            if len(qpp_vals) >= 2:
                tau, _ = kendalltau(qpp_vals, ap_vals)
                perquery_tau[qid][i] = tau

    return perquery_tau

def evaluate_all_runs(path, qrels_file, metric):
    """
    Evaluates all TREC result files in a folder and returns per-query metric scores.

    Parameters:
    - results_folder (str): Path to folder containing .res files.
    - qrels_file: Qrels file
    - metric: pyterrier.measures metric (default: AP with rel >= 2).

    Returns:
    - dict: Nested dict {query_id: {model_name: score, ...}, ...}
    """
    # Final nested dict: {qid: {model_name: score}}
    all_results = {}
    qrels = pt.io.read_qrels(qrels_file)

    files = []
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".res")]
    elif os.path.isfile(path) and path.endswith(".res"):
        files = [path]
    else:
        raise ValueError(f"Invalid path: {path} is neither a .res file nor a directory containing .res files.")
    

    for filename in files:            
        model_name = os.path.basename(filename).replace(".res", "")

        # Load run as DataFrame
        run_df = pd.read_csv(filename, 
                     sep='\\s+', 
                     names=["qid", "iter", "docno", "rank", "score", "runid"])

        # Evaluate per query
        perquery_results = pt.Evaluate(run_df, qrels, metrics=[metric], perquery=True)
        perquery_df = pd.DataFrame.from_dict(perquery_results, orient='index')
        
        # Fill into nested dictionary
        for qid, row in perquery_df.iterrows():
            metric_name = perquery_df.columns[0]
            score = row[metric_name]
            all_results.setdefault(str(qid), {})[model_name] = score

    return all_results

def load_qpp_estimates(path):
    """
    Loads QPP estimates for all queries and IR models from a folder of .qpp files.

    Each .qpp file is named after an IR model (e.g., 'bm25.qpp') and contains:
    query_id \t qpp1 \t qpp2 \t ... \t qppN

    Returns:
    - dict of {query_id: {model_name: [qpp1, qpp2, ...]}}
    """
    qpp_data = {}

    files = []
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".qpp")]
    elif os.path.isfile(path) and path.endswith(".res"):
        files = [path + ".qpp"]
    else:
        raise ValueError(f"Invalid path: {path} is neither a .res file nor a directory containing .res files.")
    
    for filename in files:
        model_name = os.path.basename(filename).replace(".res.qpp", "")        

        with open(filename, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue

                qid = parts[0]
                try:
                    preds = [float(x) for x in parts[1:]]
                except ValueError:
                    continue  # skip malformed line

                qpp_data.setdefault(qid, {})[model_name] = preds

    return qpp_data

def evaulate_all_qpp(qpp_estimates, results):
    tau_scores = compute_perquery_kendall_multiqpp(qpp_estimates, results)
    qpp_model_avgs = defaultdict(list)
    for qid in tau_scores:
        for i, tau in tau_scores[qid].items():
            if tau is not None:
                qpp_model_avgs[i].append(tau)
    
    mean_local_taus = []
    for i in sorted(qpp_model_avgs):
        mean_tau = np.mean(qpp_model_avgs[i])
        mean_local_taus.append(mean_tau)
        print(f"[QPP-eval (correlation across rankers averaged over queries)] QPP model {i}: mean Kendall's tau = {mean_tau:.4f}")    

    standard_tau = compute_modelwise_kendall_multiqpp(qpp_estimates, results)
    
    # Average Kendall's tau per QPP model (averaged over IR models)
    qpp_model_avgs = defaultdict(list)
    for model in standard_tau:
        for i, tau in standard_tau[model].items():
            if tau is not None:
                qpp_model_avgs[i].append(tau)
    
    mean_taus = []
    for i in sorted(qpp_model_avgs):
        mean_tau = np.mean(qpp_model_avgs[i])
        mean_taus.append(mean_tau)
        print(f"[QPP-eval (correlation across queries averaged over rankers)] QPP model {i}: mean Kendall's tau = {mean_tau:.4f}")
    
    global_tau = compute_global_kendall_multiqpp(qpp_estimates, results)
    
    for i in sorted(global_tau):
        print(f"[QPP-eval (correlation across each query-ranker pair)] QPP model {i}: Kendall's tau = {global_tau[i]:.4f}")    


import argparse

# Assuming these are already defined elsewhere in your code:
# - evaluate_all_runs
# - load_qpp_estimates
# - evaulate_all_qpp

def main():
    parser = argparse.ArgumentParser(description="Precise QPP Evaluator.")
    parser.add_argument("--respath", type=str, required=True, help="Path to .res file or directory of .res files")
    parser.add_argument("--qrels", type=str, required=True, help="Path to TREC qrels file")

    args = parser.parse_args()

    results = evaluate_all_runs(args.respath, args.qrels, metric=AP(rel=2))
    qpp_estimates = load_qpp_estimates(args.respath)
    
    evaulate_all_qpp(qpp_estimates, results)

if __name__ == "__main__":
    main()

