""" PoSeiDon dataset wrapper as PyG.dataset.
Author: Hongwei Jin <jinh@anl.gov>
License: TBD
"""
from utils import *
import glob
import json
import os.path as osp
import random
import shutil
import pickle
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, Data, InMemoryDataset

from utils import create_dir
from sklearn.model_selection import train_test_split
torch.manual_seed(0)
np.random.seed(0)

class PSD_Dataset(InMemoryDataset):
    """ New normalizing process """

    def __init__(self,
                 root="./",
                 name="1000genome",
                 use_node_attr=True,
                 use_edge_attr=False,
                 force_reprocess=False,
                 node_level=False,
                 binary_labels=False,
                 normalize=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        """ Customized dataset for PoSeiDon graphs.
        Args:
            root (str, optional): Root of the processed path. Defaults to "./".
            name (str, optional): Name of the workflow type. Defaults to "1000genome".
            use_node_attr (bool, optional): Use node attributes. Defaults to False.
            use_edge_attr (bool, optional): Use edge attributes. Defaults to False.
            force_reprocess (bool, optional): Force to reprocess. Defaults to False.
            node_level (bool, optional): Process as node level graphs if `True`. Defaults to False.
            transform (callable, optional): Transform function to process. Defaults to None.
            pre_transform (callable, optional): Pre_transform function. Defaults to None.
            pre_filter (callable, optional): Pre filter function. Defaults to None.
        """
        self.name = name.lower()
        # self.anomaly_type = anomaly_type.lower()
        # assert self.anomaly_type in ["all", "cpu", "hdd", "loss"]
        # self.attr_options = attr_options.lower()
        # assert self.attr_options in ["s1", "s2", "s3"]
        # force to reprocess again
        self.force_reprocess = force_reprocess
        self.node_level = node_level
        self.use_node_attr = use_node_attr
        self.use_edge_attr = use_edge_attr
        self.binary_labels = binary_labels
        self.normalize = normalize
        if self.force_reprocess:
            SAVED_PATH = osp.join(osp.abspath(root), "processed", self.name)
            SAVED_FILE = f'{SAVED_PATH}/{self.__class__.__name__}_binary_{self.binary_labels}_node_{self.node_level}.pt'
            if osp.exists(SAVED_FILE):
                # shutil.rmtree(SAVED_FILE)
                os.remove(SAVED_FILE)

        # load data if processed
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        r"""The name of the files in the :obj:`self.processed_dir` folder that
        must be present in order to skip processing.
        Returns:
            list: List of file names.
        """
        SAVED_PATH = osp.join(osp.abspath(self.root), "processed", self.name)
        create_dir(SAVED_PATH)
        return [f'{SAVED_PATH}/{self.__class__.__name__}_binary_{self.binary_labels}_node_{self.node_level}.pt',
        f'{SAVED_PATH}/{self.__class__.__name__}_binary_{self.binary_labels}_node_{self.node_level}.pkl']

    @property
    def num_node_attributes(self):
        """ Number of node features. """
        if self.data.x is None:
            return 0
        return self.data.x.shape[1] - self.num_node_labels

    @property
    def num_node_labels(self):
        """ Number of node labels.
        Returns:
            int: 3. (auxiliary, compute, transfer).
        """
        return 3

    @property
    def num_edge_attributes(self):
        raise NotImplementedError

    def process(self):
        """ Process the raw files, and save to processed files. """
        ''' process adj file '''
        adj_folder = osp.join(osp.dirname(osp.abspath(__file__)), "", "adjacency_list_dags")
        data_folder = osp.join(osp.dirname(osp.abspath(__file__)), "", "data")
        if self.name == "all":
            # TODO: process the entire graphs
            pass
        else:
            adj_file = osp.join(adj_folder, f"{self.name.replace('-', '_')}.json")
        d = json.load(open(adj_file))

        # build dict of nodes
        nodes = {k: i for i, k in enumerate(d.keys())}

        # clean up with no timestamps in nodes
        nodes = {}
        for i, k in enumerate(d.keys()):
            if k.startswith("create_dir_") or k.startswith("cleanup_"):
                k = k.split("-")[0]
                nodes[k] = i
            else:
                nodes[k] = i

        edges = []
        for u in d:
            for v in d[u]:
                if u.startswith("create_dir_") or u.startswith("cleanup_"):
                    u = u.split("-")[0]
                if v.startswith("create_dir_") or v.startswith("cleanup_"):
                    v = v.split("-")[0]
                edges.append((nodes[u], nodes[v]))

        # convert the edge_index with dim (2, E)
        edge_index = torch.tensor(edges).T

        features = ['auxiliary', 'compute', 'transfer'] + \
            ['is_clustered', 'ready', 'pre_script_start',
             'pre_script_end', 'submit', 'execute_start', 'execute_end',
             'post_script_start', 'post_script_end', 'wms_delay', 'pre_script_delay',
             'queue_delay', 'runtime', 'post_script_delay', 'stage_in_delay',
             'stage_in_bytes', 'stage_out_delay', 'stage_out_bytes', 'kickstart_executables_cpu_time',
             'kickstart_status', 'kickstart_executables_exitcode']
        data_list = []
        feat_list = []
        n = len(nodes)
        for filename in glob.glob(f"{data_folder}/*/{self.name.replace('_', ' - ')}*.csv"):
            # process labels according to classes
            if self.binary_labels:
                if "normal" in filename:
                    y = torch.tensor([0]) if not self.node_level else torch.tensor([0] * n)
                else:
                    y = torch.tensor([1]) if not self.node_level else torch.tensor([1] * n)
            else:
                if "normal" in filename:
                    y = torch.tensor([0]) if not self.node_level else torch.tensor([0] * n)
                elif "cpu" in filename:
                    y = torch.tensor([1]) if not self.node_level else torch.tensor([1] * n)
                elif "hdd" in filename:
                    y = torch.tensor([2]) if not self.node_level else torch.tensor([2] * n)
                elif "loss" in filename:
                    y = torch.tensor([3]) if not self.node_level else torch.tensor([3] * n)

            df = pd.read_csv(filename, index_col=[0])

            # handle missing features
            if "kickstart_executables_cpu_time" not in df.columns:
                continue
            # handle the nowind workflow
            if df.shape[0] != len(nodes):
                continue

            # convert type to dummy features
            df = pd.concat([pd.get_dummies(df.type), df], axis=1)
            df = df.drop(["type"], axis=1)
            df = df[features]
            df = df.fillna(0)

            # shift timestamps
            ts_anchor = df['ready'].min()
            for attr in ['ready', 'submit', 'execute_start', 'execute_end', 'post_script_start', 'post_script_end']:
                df[attr] -= ts_anchor

            # change the index same with `nodes`
            for i, node in enumerate(df.index.values):
                if node.startswith("create_dir_") or node.startswith("cleanup_"):
                    new_name = node.split("-")[0]
                    df.index.values[i] = new_name

            x = torch.tensor(df.to_numpy(), dtype=torch.float32)
            feat_list.append(df.to_numpy())
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        # normalize across jobs
        # backend: numpy
        if self.normalize:
            all_feat = np.array(feat_list)
            # v_min = all_feat.min(axis=1, keepdims=True)
            # v_max = all_feat.max(axis=1, keepdims=True)
            v_min = np.concatenate(all_feat).min()
            v_max = np.concatenate(all_feat).max()
            norm_feat = (all_feat - v_min) / (v_max - v_min) + 0
            np.nan_to_num(norm_feat, 0)
            for i, x in enumerate(norm_feat):
                data_list[i].x = torch.tensor(x, dtype=torch.float32)

        pickle.dump(data_list, open(self.processed_file_names[1], "wb"))
        # backend: pytorch
        # if self.normalize:
        #     all_feat = torch.stack(feat_list)
        #     v_min = torch.min(all_feat, dim=1, keepdim=True)[0]
        #     v_max = torch.max(all_feat, dim=1, keepdim=True)[0]
        #     norm_feat = (all_feat - v_min) / (v_max - v_min)
        #     torch.nan_to_num(norm_feat, 0, 1, 0)
        #     for i, x in enumerate(norm_feat):
        #         data_list[i].x = torch.tensor(x, dtype=torch.float32)

        # Save processed data
        if self.node_level:
            data_batch = Batch.from_data_list(data_list)
            data = Data(x=data_batch.x, edge_index=data_batch.edge_index, y=data_batch.y)
            data = data if self.pre_transform is None else self.pre_transform(data)

            # NOTE: split the dataset into train/val/test as 60/20/20
            idx = np.arange(data.num_nodes)
            train_idx, test_idx = train_test_split(
                idx, train_size=0.6, random_state=0, shuffle=True, stratify=data.y.numpy())
            val_idx, test_idx = train_test_split(
                test_idx, train_size=0.5, random_state=0, shuffle=True, stratify=data.y.numpy()[test_idx])

            data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.train_mask[train_idx] = 1

            data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.val_mask[val_idx] = 1

            data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.test_mask[test_idx] = 1
            torch.save(self.collate([data]), self.processed_paths[0])
        else:
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
            # TODO: add data_loader to the fixed split of dataset

    def __repr__(self):
        return f'{self.name}({len(self)}) node_level {self.node_level} binary_labels {self.binary_labels}'


class Merge_PSD_Dataset(InMemoryDataset):
    def __init__(self, root="./",
                 node_level=True,
                 binary_labels=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None) -> None:
        self.name = "all"
        self.root = root
        self.node_level = node_level
        self.binary_labels = binary_labels
        workflows = ["1000genome",
                     "nowcast-clustering-8",
                     "nowcast-clustering-16",
                     "wind-clustering-casa",
                     "wind-noclustering-casa"]
        # check all data are available
        for w in workflows:
            dataset = PSD_Dataset(name=w, node_level=self.node_level, binary_labels=self.binary_labels)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        r"""The name of the files in the :obj:`self.processed_dir` folder that
        must be present in order to skip processing.
        Returns:
            list: List of file names.
        """
        SAVED_PATH = osp.join(osp.abspath(self.root), "processed", self.name)
        create_dir(SAVED_PATH)
        return [f'{SAVED_PATH}/{self.__class__.__name__}_binary_{self.binary_labels}_node_{self.node_level}.pt',
                f'{SAVED_PATH}/{self.__class__.__name__}_binary_{self.binary_labels}_node_{self.node_level}.pkl']

    def process(self):
        """ process """
        data_list = []
        for wn in ["1000genome",
                   "nowcast-clustering-8",
                   "nowcast-clustering-16",
                   "wind-clustering-casa",
                   "wind-noclustering-casa"]:
            wn_path = osp.join(osp.abspath(self.root), "processed", wn)
            subdata_list = pickle.load(
                open(
                    f'{wn_path}/PSD_Dataset_binary_{self.binary_labels}_node_{self.node_level}.pkl',
                    'rb'))
            data_list += subdata_list
        pickle.dump(data_list, open(self.processed_file_names[1], "wb"))

        if self.node_level:
            data_batch = Batch.from_data_list(data_list)
            data = Data(x=data_batch.x, edge_index=data_batch.edge_index, y=data_batch.y)
            data = data if self.pre_transform is None else self.pre_transform(data)

            # NOTE: split the dataset into train/val/test as 60/20/20
            idx = np.arange(data.num_nodes)
            train_idx, test_idx = train_test_split(
                idx, train_size=0.6, random_state=0, shuffle=True, stratify=data.y.numpy())
            val_idx, test_idx = train_test_split(
                test_idx, train_size=0.5, random_state=0, shuffle=True, stratify=data.y.numpy()[test_idx])

            data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.train_mask[train_idx] = 1

            data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.val_mask[val_idx] = 1

            data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.test_mask[test_idx] = 1
            torch.save(self.collate([data]), self.processed_paths[0])
        else:
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return f'{self.name}({len(self)}) node_level {self.node_level} binary_labels {self.binary_labels}'

def sava_data_14_features(folder):
    Total_Data = []
    for flag in ['genome', 'nowcluster_16',\
         'casa_nowcast_8', 'wind_clustering', 'wind_noclustering']:
        graphs = load_14_features_data(flag)
        ## Dumped into the pickle file.
        import pickle
        with open(folder+'graph_'+str(flag)+'.pkl','wb') as f:
            pickle.dump(graphs, f)
        Total_Data.extend(graphs)
    with open(folder+'graph_all.pkl','wb') as f:
            pickle.dump(Total_Data, f)



def process_data(graphs, drop_columns, flag = '22'):## Now, preprocess that for the columns I specify
    classes = {"normal": 0}
    counter = 1
    for d in os.listdir("Raw_data/data"):
        d = d.split("_")[0]
        if d in classes: continue
        classes[d] = 1
        counter += 1
    if flag == '22':
        columns_full= ['type', 'is_clustered', 'ready',
            'submit', 'execute_start', 'execute_end', 'post_script_start',
            'post_script_end', 'wms_delay', 'pre_script_delay', 'queue_delay',
            'runtime', 'post_script_delay', 'stage_in_delay', 'stage_out_delay',
            'stage_in_bytes', 'stage_out_bytes', 'kickstart_user', 'kickstart_site', 
            'kickstart_hostname', 'kickstart_transformations', 'kickstart_executables',
            'kickstart_executables_argv', 'kickstart_executables_cpu_time', 'kickstart_status',
            'kickstart_executables_exitcode']
        number_data=[]
        flat_list = [item  for sublist in graphs for item in sublist['x']]
        number_data = [len(sublist['x']) for sublist in graphs]
        df = pd.DataFrame(flat_list, columns=columns_full)
        df = df.drop(drop_columns, axis=1)
        from sklearn.preprocessing import LabelEncoder
        labelencoder = LabelEncoder()
        for string in ['type', 'kickstart_user', 'kickstart_site',\
                    'kickstart_hostname', 'kickstart_transformations']:
                df[string] =labelencoder.fit_transform(df[string].astype(str))
    else:
        number_data=[]
        flat_list = [item  for sublist in graphs for item in sublist['x']]
        number_data = [len(sublist['x']) for sublist in graphs]
        df = pd.DataFrame(flat_list)
    
    array = df.to_numpy().astype('float64')
    prev = 0
    nexts= 0
    gx = array
    v_min, v_max = gx.min(), gx.max()
    gx = (gx - v_min)/(v_max - v_min)
    # print(gx.min(), gx.max())
    from sklearn.preprocessing import StandardScaler
    scaler =  StandardScaler()
    gx = scaler.fit_transform(gx)
    for i in range(len(graphs)):
        nexts+=number_data[i]
        graphs[i]['x']= array[prev:nexts,:]
        prev+=number_data[i]
    return graphs


def convert_tensors(graphs):
    y_list = []
    for gr in graphs:
        y_list.append(gr['y'])
    datasets=[]
    from sklearn.preprocessing import StandardScaler
    scaler =  StandardScaler()
    for element in graphs:
        gx = torch.from_numpy(element['x'] )
        ge =torch.tensor(np.array(element['edge_index']) ).T
        gy =torch.tensor(np.array(element['y']).reshape([-1]))
        if gx.shape[0] >0 :
            datasets.append( Data(x=gx, edge_index=ge, y=gy) )
    
    print('====================')
    print(f'Number of graphs: {len(datasets)}')
    # print(f'Number of features: {dataset.num_features}')
    # print(f'Number of classes: {dataset.num_classes}')
    data = datasets[0]  # Get the first graph object.
    print()
    print(data)
    print('=============================================================')
    # Gather some statistics about the first graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
    return datasets

def convert_dataloader(datasets,  batch_size):
    random.shuffle(datasets)
    train_dataset = datasets[: int(len(datasets)*0.60) ]
    valid_dataset = datasets[int(len(datasets)*0.60) : int(len(datasets)*0.80) ]
    test_dataset = datasets[int(len(datasets)*0.80):]
    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader



def sava_data_22_features(folder):
    Tota_data=[]
    classes = {"normal": 0}
    counter = 1
    for d in os.listdir("Raw_data/data"):
        d = d.split("_")[0]
        if d in classes: continue
        classes[d] = counter
        counter += 1
    columns_full= ['type', 'is_clustered', 'ready',
        'submit', 'execute_start', 'execute_end', 'post_script_start',
        'post_script_end', 'wms_delay', 'pre_script_delay', 'queue_delay',
        'runtime', 'post_script_delay', 'stage_in_delay', 'stage_out_delay',
        'stage_in_bytes', 'stage_out_bytes', 'kickstart_user', 'kickstart_site', 
        'kickstart_hostname', 'kickstart_transformations', 'kickstart_executables',
        'kickstart_executables_argv', 'kickstart_executables_cpu_time', 'kickstart_status',
        'kickstart_executables_exitcode']


    #######################################
    ## Get genome
    graphs = genome_graphs(total_columns = columns_full, classes=classes)
    print(len(graphs))
    ## Dumped into the pickle file.
    import pickle
    with open(folder+'/graph_genome.pkl','wb') as f:
        pickle.dump(graphs, f)
    Tota_data.extend(graphs)
    print(len(Tota_data))

    #######################################
    ## Get nowcluster 16
    graphs = nowcluster_16(total_columns = columns_full, classes=classes)
    print(len(graphs))
    ## Dumped into the pickle file.
    import pickle
    with open(folder+'/graph_nowcluster_16.pkl','wb') as f:
        pickle.dump(graphs, f)
    print(len(Tota_data))
    Tota_data.extend(graphs)

    #######################################
    ## Get nowcluster 8
    graphs = nowcluster_8(total_columns = columns_full, classes=classes)
    print(len(graphs))
    ## Dumped into the pickle file.
    import pickle
    with open(folder+'/graph_nowcluster_8.pkl','wb') as f:
        pickle.dump(graphs, f)
    print(len(Tota_data))
    Tota_data.extend(graphs)

    #######################################
    ## Get wind no clustering
    graphs =  wind_no_clustering(total_columns = columns_full, classes=classes)
    print(len(graphs))
    ## Dumped into the pickle file.
    import pickle
    with open(folder+'/graph_wind_no_clust.pkl','wb') as f:
        pickle.dump(graphs, f)
    print(len(Tota_data))
    Tota_data.extend(graphs)

    #######################################
    ## Get wind clustering
    graphs =  wind_clustering(total_columns = columns_full, classes=classes)
    print(len(graphs))
    ## Dumped into the pickle file.
    import pickle
    with open(folder+'/graph_wind_clust.pkl','wb') as f:
        pickle.dump(graphs, f)
    Tota_data.extend(graphs)
    print(len(Tota_data))
    import pickle
    with open(folder+'/graph_all.pkl','wb') as f:
        pickle.dump(Tota_data, f)


