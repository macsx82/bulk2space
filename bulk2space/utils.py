import pandas as pd
import scanpy
import time
from collections import defaultdict
import numpy as np
from scipy.optimize import nnls
from pyarrow import csv


def load_data(input_bulk_path,
              input_sc_data_path,
              input_sc_meta_path,
              input_st_data_path="",
              input_st_meta_path=""):
    input_sc_meta_path = input_sc_meta_path
    input_sc_data_path = input_sc_data_path
    input_bulk_path = input_bulk_path
    input_st_meta_path = input_st_meta_path
    input_st_data_path = input_st_data_path
    print("loading data......")
    #add some time check to see improvements and for debugging
    start_sc_load_time = time.time()
    input_data = {}
    # load sc_meta.csv file, containing two columns of cell name and cell type
    input_data["input_sc_meta"] = pd.read_csv(input_sc_meta_path, index_col=0)
    # load sc_data.csv file, containing gene expression of each cell
    # input_sc_data = pd.read_csv(input_sc_data_path, index_col=0)
    # use the super fast pyarrow to read the table and to convert it to pd df
    input_sc_data = csv.read_csv(input_sc_data_path, read_options=csv.ReadOptions(block_size=1e9)).to_pandas()
    # add index to dataframe
    input_sc_data.set_index('', inplace=True)

    input_data["sc_gene"] = input_sc_data._stat_axis.values.tolist()
    
    #add some time check to see improvements and for debugging
    end_sc_load_time = time.time()
    elapsed_sc_load_time=end_sc_load_time-start_sc_load_time
    print(f"SC reference data loaded in : {elapsed_sc_load_time} sec")
    
    # load bulk.csv file, containing one column of gene expression in bulk
    input_bulk = pd.read_csv(input_bulk_path, index_col=0)
    input_data["bulk_gene"] = input_bulk._stat_axis.values.tolist()
    # filter overlapping genes.
    input_data["intersect_gene"] = list(set(input_data["sc_gene"]).intersection(set(input_data["bulk_gene"])))
    input_data["input_sc_data"] = input_sc_data.loc[input_data["intersect_gene"]]
    input_data["input_bulk"] = input_bulk.loc[input_data["intersect_gene"]]
   # load st_meta.csv and st_data.csv, containing coordinates and gene expression of each spot respectively.
    if not input_st_data_path :
        print("No spatial reference provided")
    else :
        input_data["input_st_meta"] = pd.read_csv(input_st_meta_path, index_col=0)
        input_data["input_st_data"] = pd.read_csv(input_st_data_path, index_col=0)
    print("load data done!")
    
    return input_data

# function to add a different bulk dataset to an existing loaded dataset created with load_data function
def load_bulk_data(input_bulk_path,existing_input_data):
    input_bulk_path = input_bulk_path
    print("loading bulk data......")
    #add some time check to see improvements and for debugging
    start_sc_load_time = time.time()
    #we want to add a different bulk dataset to the existing input data, to perform the gene intersection since it could be useful
    input_sc_data=existing_input_data["input_sc_data"]
    input_data = existing_input_data
    
    # load bulk.csv file, containing one column of gene expression in bulk
    input_bulk = pd.read_csv(input_bulk_path, index_col=0)
    input_data["bulk_gene"] = input_bulk._stat_axis.values.tolist()
    
    #add some time check to see improvements and for debugging
    end_sc_load_time = time.time()
    elapsed_sc_load_time=end_sc_load_time-start_sc_load_time
    print(f"Bulk data loaded in : {elapsed_sc_load_time} sec")
    
    # filter overlapping genes.
    input_data["intersect_gene"] = list(set(input_data["sc_gene"]).intersection(set(input_data["bulk_gene"])))
    input_data["input_sc_data"] = input_sc_data.loc[input_data["intersect_gene"]]
    input_data["input_bulk"] = input_bulk.loc[input_data["intersect_gene"]]
   
    print("load data done!")
    
    return input_data

# define how many cells we will output (ratio_num parameter - fraction of the total numebr of cells in the reference)
# and the label that are represented in the bulk RNA data 
def data_process(data, top_marker_num, ratio_num):
    # marker used
    sc = scanpy.AnnData(data["input_sc_data"].T)
    sc.obs = data["input_sc_meta"][['Cell_type']]
    #get the marker genes for each cluster, using the info in the 'Cell_type' label
    scanpy.tl.rank_genes_groups(sc, 'Cell_type', method='wilcoxon')
    marker_df = pd.DataFrame(sc.uns['rank_genes_groups']['names']).head(top_marker_num)
    #now get the unique list of markes
    marker_array = np.array(marker_df)
    marker_array = np.ravel(marker_array)
    marker_array = np.unique(marker_array)
    marker = list(marker_array)
    #get for all cells the markers in the unique list, so to work only on the genes that better define each cluster group
    sc_marker = data["input_sc_data"].loc[marker, :]
    #get the same markers from the bulk data
    bulk_marker = data["input_bulk"].loc[marker]

    #  Data processing
    breed = data["input_sc_meta"]['Cell_type']
    breed_np = breed.values
    breed_set = set(breed_np)
    id2label = sorted(list(breed_set))  # List of breed
    label2id = {label: idx for idx, label in enumerate(id2label)}  # map breed to breed-id

    cell2label = dict()  # map cell-name to breed-id
    label2cell = defaultdict(set)  # map breed-id to cell-names
    for row in data["input_sc_meta"].itertuples():
        cell_name = getattr(row, 'Cell')
        cell_type = label2id[getattr(row, 'Cell_type')]
        cell2label[cell_name] = cell_type
        label2cell[cell_type].add(cell_name)

    label_devide_data = dict()
    for label, cells in label2cell.items():
        label_devide_data[label] = sc_marker[list(cells)]

    #calculate the mean expression for each marker gene across all cells in the cluster
    single_cell_splitby_breed_np = {}
    for key in label_devide_data.keys():
        single_cell_splitby_breed_np[key] = label_devide_data[key].values  # [gene_num, cell_num]
        single_cell_splitby_breed_np[key] = single_cell_splitby_breed_np[key].mean(axis=1)

    max_decade = len(single_cell_splitby_breed_np.keys())
    single_cell_matrix = []

    for i in range(max_decade):
        single_cell_matrix.append(single_cell_splitby_breed_np[i].tolist())

    single_cell_matrix = np.array(single_cell_matrix)
    single_cell_matrix = np.transpose(single_cell_matrix)  # (gene_num, label_num)

    #now get the values for the bulk data
    bulk_marker = bulk_marker.values  # (gene_num, 1)
    bulk_rep = bulk_marker.reshape(bulk_marker.shape[0], )

    # calculate celltype ratio in each spot by NNLS (NonNegativeLeastSquares): we are doing a regression to see if the bulk data and each cell type are correlated
    ratio = nnls(single_cell_matrix, bulk_rep)[0]
    ratio = ratio / sum(ratio)

    ratio_array = np.round(ratio * data["input_sc_meta"].shape[0] * ratio_num)
    ratio_list = [r for r in ratio_array]

    cell_target_num = dict(zip(id2label, ratio_list))

    return cell_target_num



