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
    if file_name is None:
    	print(pred_edges)
    else:
    	pred_edges.to_csv(file_name, sep='\t', header=None, index=None) 
                
                
def estimate_degradation_rates(TS_data,time_points):
    
    """
    For each gene, the degradation rate is estimated by assuming that the gene expression x(t) follows:
    x(t) =  A exp(-alpha * t) + C_min,
    between the highest and lowest expression values.
    C_min is set to the minimum expression value over all genes and all samples.
    """
    
    ngenes = TS_data[0].shape[1]
    nexp = len(TS_data)
    
    C_min = TS_data[0].min()
    if nexp > 1:
        for current_timeseries in TS_data[1:]:
            C_min = min(C_min,current_timeseries.min())
    
    alphas = np.zeros((nexp,ngenes))
    
    for (i,current_timeseries) in enumerate(TS_data):
        current_time_points = time_points[i]
        
        for j in range(ngenes):
            
            idx_min = np.argmin(current_timeseries[:,j])
            idx_max = np.argmax(current_timeseries[:,j])
            
            xmin = current_timeseries[idx_min,j]
            xmax = current_timeseries[idx_max,j]
            
            tmin = current_time_points[idx_min]
            tmax = current_time_points[idx_max]
            
            xmin = max(xmin-C_min,1e-6)
            xmax = max(xmax-C_min,1e-6)
                
            xmin = np.log(xmin)
            xmax = np.log(xmax)
            
            alphas[i,j] = (xmax - xmin) / abs(tmin - tmax)
                
    alphas = alphas.max(axis=0)
 
    return alphas               



def get_importances(TS_data, time_points, alpha="from_data",  SS_data=None, gene_names=None,regulators='all', param={}):
    

    time_start = time.time()

    ngenes = TS_data[0].shape[1]
    
    if alpha is "from_data":
        alphas = estimate_degradation_rates(TS_data,time_points)
    else:
        alphas = [alpha] * ngenes
    
    # Get the indices of the candidate regulators
    idx = [i for i, gene in enumerate(gene_names) if gene in regulators]
    
    # Learn an ensemble of trees for each target gene, and compute scores for candidate regulators
    VIM = np.zeros((ngenes,ngenes))
    
    for i in range(ngenes):
        input_idx = idx.copy()
        if i in input_idx:
            input_idx.remove(i)
        #print('Gene %d/%d...' % (i+1,ngenes))      
        vi = get_importances_single(TS_data, time_points, alphas[i], input_idx, i, SS_data, param)
        VIM[i,:] = vi

 
    time_end = time.time()
    print("Elapsed time: %.2f seconds" % (time_end - time_start))

    return VIM
    


def get_importances_single(TS_data, time_points, alpha, input_idx, output_idx, SS_data, param):

    h = 1

    ngenes = TS_data[0].shape[1]
    nexp = len(TS_data)
    nsamples_time = sum([expr_data.shape[0] for expr_data in TS_data]) 
    ninputs = len(input_idx)

    # Construct training sample 

    # Time-series data
    input_matrix_time = np.zeros((nsamples_time-h*nexp,ninputs))
    output_vect_time = np.zeros(nsamples_time-h*nexp)

    nsamples_count = 0
    for (i,current_timeseries) in enumerate(TS_data):
        current_time_points = time_points[i]
        npoints = current_timeseries.shape[0]
        time_diff_current = current_time_points[h:] - current_time_points[:npoints-h]
        current_timeseries_input = current_timeseries[:npoints-h,input_idx]
        current_timeseries_output = (current_timeseries[h:,output_idx] - current_timeseries[:npoints-h,output_idx]) / time_diff_current + alpha*current_timeseries[:npoints-h,output_idx]
        nsamples_current = current_timeseries_input.shape[0]
        input_matrix_time[nsamples_count:nsamples_count+nsamples_current,:] = current_timeseries_input
        output_vect_time[nsamples_count:nsamples_count+nsamples_current] = current_timeseries_output
        nsamples_count += nsamples_current


    # Steady-state data
    if SS_data is not None: 
        input_matrix_steady = SS_data[:,input_idx]
        output_vect_steady = SS_data[:,output_idx] * alpha
    
        # Concatenation
        input_all = np.vstack([input_matrix_steady,input_matrix_time])
        output_all = np.concatenate((output_vect_steady,output_vect_time))
    else:
        input_all = input_matrix_time
        output_all = output_vect_time

    treeEstimator = XGBRegressor(**param)

    # Learn ensemble of trees
    treeEstimator.fit(input_all,output_all)
    
    # Compute importance scores
    feature_importances = treeEstimator.feature_importances_
    vi = np.zeros(ngenes)
    vi[input_idx] = feature_importances
       
    return vi

def get_scores(VIM, gold_edges, gene_names, regulators):

    idx = [i for i, gene in enumerate(gene_names) if gene in regulators]
    pred_edges = [(gene_names[j], gene_names[i], score) for (i, j), score in np.ndenumerate(VIM) if i != j and j in idx]
    pred_edges = pd.DataFrame(pred_edges)
    pred_edges = pred_edges.iloc[:100000]
    final = pd.merge(pred_edges, gold_edges, on=[0, 1], how='inner')
    auroc = roc_auc_score(final['2_y'], final['2_x'])
    aupr = average_precision_score(final['2_y'], final['2_x'])
    
    return auroc, aupr
        
        
        
