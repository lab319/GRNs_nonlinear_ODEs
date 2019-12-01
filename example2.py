from xgbgrn_2 import *

***
This example infers gene regulatory networks from the gene expression data of Escherichia coli under cold environment.
***

coldstress_data = pd.read_csv('Ecoli/coldstress.txt', '\t')
gene_names = list(coldstress_data.columns[1:])
coldstress_data = coldstress_data.values[:, 1:]

gold_edges = pd.read_csv("Ecoli/DREAM5_NetworkInference_GoldStandard_Network3.tsv", '\t', header=None)
regulators = [i for i in set(gold_edges[0]) if i in gene_names]

xgb_param = dict(max_depth=5, subsample=0.86, n_jobs=-1, n_estimators=120)
VIM = get_importances(coldstress_data, gene_names=gene_names,regulators=regulators, param=xgb_param)

auroc, aupr = get_scores(VIM, gold_edges, gene_names, regulators)

print("AUROC:", auroc, "AUPR:", aupr)