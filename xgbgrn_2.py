from xgboost import XGBRegressor
import numpy as np
import pandas as pd
import time
from sklearn.metrics import roc_auc_score, average_precision_score


def get_links(VIM, gene_names, regulators, sort=True, file_name=None):
    
    idx = [i for i, gene in enumerate(gene_names) if gene in regulators]
    pred_edges = [(gene_names[j], gene_names[i], score) for (i, j),score in np.ndenumerate(VIM) if i!=j and j in idx]
    pred_edges = pd.DataFrame(pred_edges)
    if sort is True:
        pred_edges.sort_values(2, ascending=False, inplace=True)
    pred_edges = pred_edges.iloc[:100000]
    if file_name is None:
    	print(pred_edges)
    else:
    	pred_edges.to_csv(file_name, sep='\t', header=None, index=None) 
                
                


def get_importances(expr_data, gene_names, regulators, param={}):
    

    time_start = time.time()

    ngenes = expr_data.shape[1]
    
    # Get the indices of the candidate regulators
    idx = [i for i, gene in enumerate(gene_names) if gene in regulators]
    
    # Learn an ensemble of trees for each target gene, and compute scores for candidate regulators
    VIM = np.zeros((ngenes,ngenes))
    
    for i in range(ngenes):
        input_idx = idx.copy()
        if i in input_idx:
            input_idx.remove(i)   
        vi = get_importances_single(expr_data,i,input_idx, param)
        VIM[i,:] = vi
 
    time_end = time.time()
    print("Elapsed time: %.2f seconds" % (time_end - time_start))

    return VIM
    


def get_importances_single(expr_data,output_idx,input_idx, param):
    
    ngenes = expr_data.shape[1]
    
    # Expression of target gene
    output = expr_data[:,output_idx]
    
    # Normalize output data
    output = output / np.std(output)

    expr_data_input = expr_data[:,input_idx]
    treeEstimator = XGBRegressor(**param)

    # Learn ensemble of trees
    treeEstimator.fit(expr_data_input,output)
    
    # Compute importance scores
    feature_importances = treeEstimator.feature_importances_
    vi = np.zeros(ngenes)
    vi[input_idx] = feature_importances
       
    return vi
	
def get_scores(VIM, gold_edges, gene_names, regulators):

    idx = [i for i, gene in enumerate(gene_names) if gene in regulators]
    pred_edges = [(gene_names[j], gene_names[i], score) for (i, j), score in np.ndenumerate(VIM) if i != j and j in idx]
    pred_edges = pd.DataFrame(pred_edges)
    pred_edges.sort_values(2, ascending=False, inplace=True)
    pred_edges = pred_edges.iloc[:100000]
    final = pd.merge(pred_edges, gold_edges, on=[0, 1], how='inner')
    auroc = roc_auc_score(final['2_y'], final['2_x'])
    aupr = average_precision_score(final['2_y'], final['2_x'])
    
    return auroc, aupr
        
        
        
