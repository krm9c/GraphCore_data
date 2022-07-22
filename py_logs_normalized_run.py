
from sched import scheduler
import torch 
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,  SAGEConv
from torch_geometric.nn import global_mean_pool
import torch.optim.lr_scheduler as lrs
import torch_geometric as pyg
import os
import pandas as pd
import numpy as np 
import random
from torch_geometric.data import Data
from dataset import PSD_Dataset, Merge_PSD_Dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch_geometric.loader import DataLoader

torch.manual_seed(12345)
random.seed(12345)


def process_data(graphs, drop_columns, flag = '22'):## Now, preprocess that for the columns I specify
    classes = {"normal": 0}
    counter = 1
    for d in os.listdir("data"):
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


class GCN(torch.nn.Module):
    def __init__(self, in_chann, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = SAGEConv(in_chann, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 4)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        # 3. Apply a final classifier
        x = F.dropout(x, p=0.07, training=self.training)
        x = self.lin(x)
        return x


def train(train_loader, model, criterion, optimizer):
    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data.x.float().to(device), data.edge_index.to(device), data.batch.to(device))  # Perform a single forward pass.
        loss = criterion(out, data.y.to(device))  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
    return loss.detach().cpu().item()

@torch.no_grad()
def test(loader, model):
    model.eval()
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x.float().to(device), data.edge_index.to(device), data.batch.to(device))  
        pred = out.argmax(dim=1).detach().cpu()  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return (correct / len(loader.dataset))  # Derive ratio of correct predictions.


def train_model(file, save_file, n_epochs, flag, learning_r = 1e-03):
    ## Dumped into the pickle file.
    import pickle
    from sklearn.model_selection import train_test_split
    if flag == '22':
        with open(file,'rb') as f:
            graphs = pickle.load(f)
        drop_columns= [ 'stage_in_bytes', 'stage_out_bytes', 'kickstart_executables','kickstart_executables_argv']
        graphs = process_data(graphs, drop_columns, flag)
        model = GCN(in_chann = 22, hidden_channels=100).float().to(device)
        dataset= convert_tensors(graphs)
        train_loader, _, test_loader = convert_dataloader(dataset, batch_size=32)
    elif flag=='new': 
        if file == "all":
            dataset = Merge_PSD_Dataset(node_level=False, binary_labels=False)
        else:
            dataset = PSD_Dataset(name=file, node_level=False, binary_labels=False).shuffle()
        n_graphs = len(dataset)
        y = dataset.data.y.numpy()
        train_idx, test_idx = train_test_split(np.arange(n_graphs), train_size=0.6, random_state=0, shuffle=True)
        val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=0, shuffle=True)
        train_loader = DataLoader(dataset[train_idx], batch_size=32)
        # val_loader = DataLoader(dataset[val_idx], batch_size=32)
        test_loader = DataLoader(dataset[test_idx], batch_size=32)
        # n_nodes = data.num_nodes
        NUM_NODE_FEATURES = dataset.num_node_features
        model = GCN(in_chann = NUM_NODE_FEATURES, hidden_channels=100).float().to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_r, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)
    Loss=np.zeros((n_epochs,1))
    Train_acc=np.zeros((n_epochs,1))
    Test_acc=np.zeros((n_epochs,1))             
    for epoch in range(1, n_epochs):
        Loss[epoch] = train(train_loader, model, criterion, optimizer)
        Train_acc[epoch] = test(train_loader, model)
        Test_acc[epoch] = test(test_loader, model)
        if epoch%10==0:
            my_lr_scheduler .step()
            print('Epoch:', epoch, 'Loss:',  Loss[epoch-1], 'Train Acc:', Train_acc[epoch], 'Test Acc:', Test_acc[epoch])
            torch.save(model.state_dict(), 'models/model_'+save_file+'_')
    np.savetxt('Logs/'+save_file+'_log.csv', np.concatenate([Loss, Train_acc, Test_acc], axis=1), delimiter=',')
    return Test_acc[epoch]

########################################
## To save the pickle files.
########################################
# print("Data with 14 features")
# sava_data_14_features('pickles_14_features')
# print("Data with 22 features")
# sava_data_22_features('pickles_22_features')
########################################


import timeit
acc = np.zeros((6,1))
time_ar = np.zeros((6,1))

lists =  ['graph_genome.pkl', 
'graph_nowcluster_16.pkl',
'graph_nowcluster_8.pkl',
'graph_wind_no_clust.pkl',
'graph_wind_clust.pkl',
'graph_all.pkl']

# set the seed
torch.manual_seed(2022)
random.seed(12345)

# load the dataset
# lists = ["1000genome", 
#         "casa_nowcast-clustering-8",
#         "casa_nowcast-clustering-16",
#         "casa_wind-clustering-casa",
#         "casa_wind-noclustering-casa",
#         "all"]

# workflows = ["1000genome",
#                 "nowcast-clustering-8",
#                 "nowcast-clustering-16",
#                 "wind-clustering-casa",
#                 "wind-noclustering-casa",
#                 "all"]

# records the time at this instant
# of the program
folder = ['pickles_22_features/']
flags = ['22']
for j, element in enumerate(folder):
    for i in range(len(lists)):
        # if flags[j] == 'new':
        #     print("The model being run", workflows[i])
        #     start = timeit.default_timer()
        #     acc[i] = train_model(element+workflows[i], workflows[i], flag = flags[j], n_epochs=2000)  
        #     end = timeit.default_timer()  
        #     # store the value
        #     time_ar[i] = end-start
        # else:
        print("The model being run", lists[i])
        start = timeit.default_timer()
        acc[i] = train_model(element+lists[i], lists[i], flag = flags[j], n_epochs=2000)  
        end = timeit.default_timer()  
        # store the value
        time_ar[i] = end-start
    print("The final accuracies", acc, time_ar)