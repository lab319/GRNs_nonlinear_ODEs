from xgbgrn import *

***
This example combines the time-series data of insilico_size10 and the steady-state data of  insilico_size10 
to infer gene regulatory networks.
***

TS_data = pd.read_csv("DREAM4/insilico_size10/insilico_size10_1_timeseries.tsv", sep='\t').values
SS_data_1 = pd.read_csv("DREAM4/insilico_size10/insilico_size10_1_knockouts.tsv", sep='\t').values
SS_data_2 = pd.read_csv("DREAM4/insilico_size10/insilico_size10_1_knockdowns.tsv", sep='\t').values

# get the steady-state data
SS_data = np.vstack([SS_data_1, SS_data_2])

i = np.arange(0, 85, 21)
j = np.arange(21, 106, 21)

# get the time-series data
TS_data = [TS_data[i:j] for (i, j) in zip(i, j)]
# get time points
time_points = [np.arange(0, 1001, 50)] * 5

ngenes = TS_data[0].shape[1]
gene_names = ['G'+str(i+1) for i in range(ngenes)]
regulators = gene_names.copy()

gold_edges = pd.read_csv("DREAM4/insilico_size10/insilico_size10_1_goldstandard.tsv", '\t', header=None)

xgb_kwargs = dict(n_estimators=398, learning_rate=0.0133, importance_type="weight", max_depth=5, n_jobs=-1)

VIM = get_importances(TS_data, time_points, alpha=0.0214, SS_data=SS_data, gene_names=gene_names,
                      regulators=regulators, param=xgb_kwargs)
auroc, aupr = get_scores(VIM, gold_edges, gene_names, regulators)
print("AUROC:", auroc, "AUPR:", aupr)