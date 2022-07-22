import os
import numpy as np
import pandas as pd
def create_dir(path):
    """ Create a dir where the processed data will be stored
    Args:
        path (str): Path to create the folder.
    """
    dir_exists = os.path.exists(path)

    if not dir_exists:
        try:
            os.makedirs(path)
            print("The {} directory is created.".format(path))
        except Exception as e:
            print("Error: {}".format(e))
            exit(-1)


def load_14_features_data(flag):
    graphs = []
    classes = {"normal": 0}
    counter = 1
    for d in os.listdir("Raw_data/data"):
        d = d.split("_")[0]
        if d in classes: continue
        classes[d] = counter
        counter += 1
        
    if flag=='genome':
        print("1000genome")
        counter = 0
        edge_index = []
        lookup = {}
        with open("Raw_data/adjacency_list_dags/1000genome.json", "r") as f:
            adjacency_list = json.load(f)
        for l in adjacency_list:
            lookup[l] = counter
            counter+=1
        for l in adjacency_list:
            for e in adjacency_list[l]:
                edge_index.append([lookup[l], lookup[e]])
        import math
        for d in os.listdir("Raw_data/data"):
            print(d)
            for f in glob.glob(os.path.join("Raw_data/data", d, "1000genome*")):
                graph = {"y": classes[d.split("_")[0]], "edge_index": edge_index, "x":[]}
                features = pd.read_csv(f, index_col=[0])
                features= features.replace('', -1, regex=True)
                # time_list
                for l in lookup:
                    if l.startswith("create_dir_") or l.startswith("cleanup_"):
                        new_l = l.split("-")[0]
                    else:
                        new_l = l
                    job_features = features[features.index.str.startswith(new_l)][['type', 'ready',
                                           'submit', 'execute_start', 'execute_end', 'post_script_start',
                                           'post_script_end', 'wms_delay', 'pre_script_delay', 'queue_delay',
                                           'runtime', 'post_script_delay', 'stage_in_delay', 'stage_out_delay']].values.tolist()[0]

                    if job_features[0]=='auxiliary':
                        job_features[0]= 0
                    if job_features[0]=='compute':
                        job_features[0]= 1
                    if job_features[0]=='transfer':
                        job_features[0]= 2

                    # print(job_features)
                    job_features = [-1 if x != x else x for x in job_features]
                    graph['x'].insert(lookup[l], job_features)
                t_list=[]
                for i in range(len(graph['x'])):
                    t_list.append(graph['x'][i][1])
                minim= min(t_list)
                for i in range(len(graph['x'])):
                    lim = graph['x'][i][1:7]
                    lim=[ v-minim for v in lim]
                    graph['x'][i][1:7]= lim
                graphs.append(graph)
    elif flag=='nowcluster_16':
        print("casa_nowcast_clustering_16")
        counter = 0
        edge_index = []
        lookup = {}
        with open("Raw_data/adjacency_list_dags/casa_nowcast_clustering_16.json", "r") as f:
            adjacency_list = json.load(f)
        for l in adjacency_list:
            lookup[l] = counter
            counter+=1
        for l in adjacency_list:
            for e in adjacency_list[l]:
                edge_index.append([lookup[l], lookup[e]])               
        for d in os.listdir("Raw_data/data"):
            print(d)
            for f in glob.glob(os.path.join("Raw_data/data", d, "nowcast-clustering-16*")):
                graph = {"y": classes[d.split("_")[0]], "edge_index": edge_index, "x":[]}
                features = pd.read_csv(f, index_col=[0])
                features= features.replace('', -1, regex=True)
                # time_list
                for l in lookup:
                    if l.startswith("create_dir_") or l.startswith("cleanup_"):
                        new_l = l.split("-")[0]
                    else:
                        new_l = l
                    job_features = features[features.index.str.startswith(new_l)][['type', 'ready',
                                           'submit', 'execute_start', 'execute_end', 'post_script_start',
                                           'post_script_end', 'wms_delay', 'pre_script_delay', 'queue_delay',
                                           'runtime', 'post_script_delay', 'stage_in_delay', 'stage_out_delay']].values.tolist()[0]
                    # print(job_features)
                    if job_features[0]=='auxiliary':
                        job_features[0]= 0
                    if job_features[0]=='compute':
                        job_features[0]= 1
                    if job_features[0]=='transfer':
                        job_features[0]= 2
                        #             print(job_features)
                    job_features = [-1 if x != x else x for x in job_features]
                    graph['x'].insert(lookup[l], job_features)
                t_list=[]
                for i in range(len(graph['x'])):
                    t_list.append(graph['x'][i][1])
                minim= min(t_list)
                for i in range(len(graph['x'])):
                    lim = graph['x'][i][1:7]
                    lim=[ v-minim for v in lim]
                    graph['x'][i][1:7]= lim            
                graphs.append(graph)
    elif flag=='casa_nowcast_8':
        print("casa_nowcast_clustering_8")
        counter = 0
        edge_index = []
        lookup = {}
        with open("Raw_data/adjacency_list_dags/casa_nowcast_clustering_8.json", "r") as f:
            adjacency_list = json.load(f)
        for l in adjacency_list:
            lookup[l] = counter
            counter+=1
        for l in adjacency_list:
            for e in adjacency_list[l]:
                edge_index.append([lookup[l], lookup[e]])     
        for d in os.listdir("Raw_data/data"):
            print(d)
            for f in glob.glob(os.path.join("Raw_data/data", d, "nowcast-clustering-8*")):
                graph = {"y": classes[d.split("_")[0]], "edge_index": edge_index, "x":[]}
                features = pd.read_csv(f, index_col=[0])
                features= features.replace('', -1, regex=True)
                # time_list
                for l in lookup:
                    if l.startswith("create_dir_") or l.startswith("cleanup_"):
                        new_l = l.split("-")[0]
                    else:
                        new_l = l
                    job_features = features[features.index.str.startswith(new_l)][['type', 'ready',
                                           'submit', 'execute_start', 'execute_end', 'post_script_start',
                                           'post_script_end', 'wms_delay', 'pre_script_delay', 'queue_delay',
                                           'runtime', 'post_script_delay', 'stage_in_delay', 'stage_out_delay']].values.tolist()[0]
                    if job_features[0]=='auxiliary':
                        job_features[0]= 0
                    if job_features[0]=='compute':
                        job_features[0]= 1
                    if job_features[0]=='transfer':
                        job_features[0]= 2
                        #             print(job_features)
                    job_features = [-1 if x != x else x for x in job_features]
                    graph['x'].insert(lookup[l], job_features)
                t_list=[]
                for i in range(len(graph['x'])):
                    t_list.append(graph['x'][i][1])
                minim= min(t_list)
                for i in range(len(graph['x'])):
                    lim = graph['x'][i][1:7]
                    lim=[ v-minim for v in lim]
                    graph['x'][i][1:7]= lim
                graphs.append(graph)
    elif flag=='wind_clustering':
        print("casa_wind_clustering")
        counter = 0
        edge_index = []
        lookup = {}
        with open("Raw_data/adjacency_list_dags/casa_wind_clustering.json", "r") as f:
            adjacency_list = json.load(f)
        for l in adjacency_list:
            lookup[l] = counter
            counter+=1
        for l in adjacency_list:
            for e in adjacency_list[l]:
                edge_index.append([lookup[l], lookup[e]])   
        for d in os.listdir("Raw_data/data"):
            print(d)
            for f in glob.glob(os.path.join("Raw_data/data", d, "wind-clustering-casa*")):
                graph = {"y": classes[d.split("_")[0]], "edge_index": edge_index, "x":[]}
                features = pd.read_csv(f, index_col=[0])
                features= features.replace('', -1, regex=True)
                # time_list
                for l in lookup:
                    if l.startswith("create_dir_") or l.startswith("cleanup_"):
                        new_l = l.split("-")[0]
                    else:
                        new_l = l
                    job_features = features[features.index.str.startswith(new_l)][['type', 'ready',
                                           'submit', 'execute_start', 'execute_end', 'post_script_start',
                                           'post_script_end', 'wms_delay', 'pre_script_delay', 'queue_delay',
                                           'runtime', 'post_script_delay', 'stage_in_delay', 'stage_out_delay']].values.tolist()[0]
                    if job_features[0]=='auxiliary':
                        job_features[0]= 0
                    if job_features[0]=='compute':
                        job_features[0]= 1
                    if job_features[0]=='transfer':
                        job_features[0]= 2
                        #             print(job_features)
                    job_features = [-1 if x != x else x for x in job_features]
                    graph['x'].insert(lookup[l], job_features)
                t_list=[]
                for i in range(len(graph['x'])):
                    t_list.append(graph['x'][i][1])
                minim= min(t_list)
                for i in range(len(graph['x'])):
                    lim = graph['x'][i][1:7]
                    lim=[ v-minim for v in lim]
                    graph['x'][i][1:7]= lim
                graphs.append(graph)
    elif flag=='wind_noclustering':
        print("casa_wind_no_clustering")
        counter = 0
        edge_index = []
        lookup = {}
        with open("Raw_data/adjacency_list_dags/casa_wind_no_clustering.json", "r") as f:
            adjacency_list = json.load(f)
        for l in adjacency_list:
            lookup[l] = counter
            counter+=1
        for l in adjacency_list:
            for e in adjacency_list[l]:
                edge_index.append([lookup[l], lookup[e]])     
        for d in os.listdir("Raw_data/data"):
            print(d)
            for f in glob.glob(os.path.join("Raw_data/data", d, "wind-noclustering-casa*")):
                if "-20200817T052029Z-" in f: continue
                graph = {"y": classes[d.split("_")[0]], "edge_index": edge_index, "x":[]}
                features = pd.read_csv(f, index_col=[0])
                features= features.replace('', -1, regex=True)
                # time_list
                for l in lookup:
                    if l.startswith("create_dir_") or l.startswith("cleanup_"):
                        new_l = l.split("-")[0]
                    else:
                        new_l = l
                    # print(len(features[features.index.str.startswith(new_l)]) )
                    #type,ready,pre_script_start,pre_script_end,submit,execute_start,
                    #execute_end,post_script_start,post_script_end,wms_delay,pre_script_delay,
                    #queue_delay,runtime,post_script_delay,stage_in_delay,stage_out_delay
                    if len(features[features.index.str.startswith(new_l)])<1:
                        print(f)
                        print(new_l)
                        continue

                    job_features = features[features.index.str.startswith(new_l)][['type', 'ready',
                                           'submit', 'execute_start', 'execute_end', 'post_script_start',
                                           'post_script_end', 'wms_delay', 'pre_script_delay', 'queue_delay',
                                           'runtime', 'post_script_delay', 'stage_in_delay', 'stage_out_delay']].values.tolist()[0]
                    if job_features[0]=='auxiliary':
                        job_features[0]= 0
                    if job_features[0]=='compute':
                        job_features[0]= 1
                    if job_features[0]=='transfer':
                        job_features[0]= 2
                        #             print(job_features)
                    job_features = [-1 if x != x else x for x in job_features]
                    graph['x'].insert(lookup[l], job_features)
                t_list=[]
                for i in range(len(graph['x'])):
                    t_list.append(graph['x'][i][1])
                minim= min(t_list)
                for i in range(len(graph['x'])):
                    lim = graph['x'][i][1:7]
                    lim=[ v-minim for v in lim]
                    graph['x'][i][1:7]= lim
                graphs.append(graph) 
    return graphs

total_columns = ['type', 'is_clustered', 'ready',
    'submit', 'execute_start', 'execute_end', 'post_script_start',
    'post_script_end', 'wms_delay', 'pre_script_delay', 'queue_delay',
    'runtime', 'post_script_delay', 'stage_in_delay', 'stage_out_delay',
    'stage_in_bytes', 'stage_out_bytes', 'kickstart_user', 'kickstart_site', 
    'kickstart_hostname', 'kickstart_transformations', 'kickstart_executables',
    'kickstart_executables_argv', 'kickstart_executables_cpu_time', 'kickstart_status',
    'kickstart_executables_exitcode']



def genome_graphs(total_columns, classes):
    print("genome")
    with open("Raw_data/adjacency_list_dags/1000genome.json", "r") as f:
        adjacency_list = json.load(f)
    counter = 0
    edge_index = []
    lookup = {}
    for l in adjacency_list:
        lookup[l] = counter
        counter+=1
    for l in adjacency_list:
        for e in adjacency_list[l]:
            edge_index.append([lookup[l], lookup[e]])
    import math
    count =0
    graphs=[]
    for d in os.listdir("Raw_data/data"):
        print(d)
        for f in glob.glob(os.path.join("Raw_data/data", d, "1000genome*")):
            if f in ["Raw_data/data/cpu_3/1000genome-20200613T072602Z-run0011.csv",
                    "Raw_data/data/hdd_100/1000genome-20200901T234910Z-run0048.csv",
                    "Raw_data/data/hdd_100/1000genome-20200901T234910Z-run0049.csv",
                    "Raw_data/data/loss_1/1000genome-20200520T215721Z-run0014.csv",
                    "Raw_data/data/norma/1000genome-20200616T174351Z-run0022.csv", 
                    "Raw_data/data/normal/1000genome-20200616T174351Z-run0022.csv"]: continue
            graph = {"y": classes[d.split("_")[0]], "edge_index": edge_index, "x":[]}
            features = pd.read_csv(f, index_col=[0])
            for l in lookup:
                if l.startswith("create_dir_") or l.startswith("cleanup_"):
                    new_l = l.split("-")[0]
                else:
                    new_l = l
                job_features = features[features.index.str.startswith(new_l)][total_columns].values.tolist()[0]
                job_features = [-1 if x != x else x for x in job_features]   
                minim = min(job_features[2:8])
                job_features[2:8] = [ j-minim for j in job_features[2:8]]
                graph['x'].insert(lookup[l], job_features)
            graphs.append(graph)
    return graphs


def wind_no_clustering(total_columns, classes):
    counter = 0
    edge_index = []
    lookup = {}
    graphs=[]
    print("casa_wind_no_clustering")
    with open("Raw_data/adjacency_list_dags/casa_wind_no_clustering.json", "r") as f:
        adjacency_list = json.load(f)
    for l in adjacency_list:
        lookup[l] = counter
        counter+=1
    for l in adjacency_list:
        for e in adjacency_list[l]:
            edge_index.append([lookup[l], lookup[e]])     
    for d in os.listdir("Raw_data/data"):
        print(d)
        for f in glob.glob(os.path.join("Raw_data/data", d, "wind-noclustering-casa*")):
            if "-20200817T052029Z-" in f: continue
            graph = {"y": classes[d.split("_")[0]], "edge_index": edge_index, "x":[]}
            features = pd.read_csv(f, index_col=[0])
            features= features.replace('', -1, regex=True)
            # time_list
            for l in lookup:
                if l.startswith("create_dir_") or l.startswith("cleanup_"):
                    new_l = l.split("-")[0]
                else:
                    new_l = l
                job_features = features[features.index.str.startswith(new_l)][total_columns].values.tolist()[0]
                job_features = [-1 if x != x else x for x in job_features]   
                minim = min(job_features[2:8])
                job_features[2:8] = [ j-minim for j in job_features[2:8]]
                graph['x'].insert(lookup[l], job_features)
            graphs.append(graph) 
    return graphs


def wind_clustering(total_columns, classes):
    graphs=[]
    print("casa_wind_clustering")
    counter = 0
    edge_index = []
    lookup = {}
    with open("Raw_data/adjacency_list_dags/casa_wind_clustering.json", "r") as f:
        adjacency_list = json.load(f)
    for l in adjacency_list:
        lookup[l] = counter
        counter+=1
    for l in adjacency_list:
        for e in adjacency_list[l]:
            edge_index.append([lookup[l], lookup[e]])   
    for d in os.listdir("Raw_data/data"):
        print(d)
        for f in glob.glob(os.path.join("Raw_data/data", d, "wind-clustering-casa*")):
            graph = {"y": classes[d.split("_")[0]], "edge_index": edge_index, "x":[]}
            features = pd.read_csv(f, index_col=[0])
            features= features.replace('', -1, regex=True)
            # print(features)
            # time_list
            for l in lookup:
                if l.startswith("create_dir_") or l.startswith("cleanup_"):
                    new_l = l.split("-")[0]
                else:
                    new_l = l
                job_features = features[features.index.str.startswith(new_l)][total_columns].values.tolist()[0]
                job_features = [-1 if x != x else x for x in job_features]   
                minim = min(job_features[2:8])
                job_features[2:8] = [ j-minim for j in job_features[2:8]]
                graph['x'].insert(lookup[l], job_features)
            graphs.append(graph)
    return graphs



def nowcluster_8(total_columns, classes):
    print("casa_nowcast_clustering_8")
    graphs = []
    counter = 0
    edge_index = []
    lookup = {}
    with open("Raw_data/adjacency_list_dags/casa_nowcast_clustering_8.json", "r") as f:
        adjacency_list = json.load(f)
    for l in adjacency_list:
        lookup[l] = counter
        counter+=1
    for l in adjacency_list:
        for e in adjacency_list[l]:
            edge_index.append([lookup[l], lookup[e]])     
    for d in os.listdir("Raw_data/data"):
        print(d)
        for f in glob.glob(os.path.join("Raw_data/data", d, "nowcast-clustering-8*")):
            graph = {"y": classes[d.split("_")[0]], "edge_index": edge_index, "x":[]}
            features = pd.read_csv(f, index_col=[0])
            features= features.replace('', -1, regex=True)
            # time_list
            for l in lookup:
                if l.startswith("create_dir_") or l.startswith("cleanup_"):
                    new_l = l.split("-")[0]
                else:
                    new_l = l
                job_features = features[features.index.str.startswith(new_l)][total_columns].values.tolist()[0]
                job_features = [-1 if x != x else x for x in job_features]   
                minim = min(job_features[2:8])
                job_features[2:8] = [ j-minim for j in job_features[2:8]]
                graph['x'].insert(lookup[l], job_features)
            graphs.append(graph)
    return graphs


def nowcluster_16(total_columns, classes):
    print("casa_nowcast_clustering_16")
    graphs = []
    counter = 0
    edge_index = []
    lookup = {}
    with open("Raw_data/adjacency_list_dags/casa_nowcast_clustering_16.json", "r") as f:
        adjacency_list = json.load(f)
    for l in adjacency_list:
        lookup[l] = counter
        counter+=1
    for l in adjacency_list:
        for e in adjacency_list[l]:
            edge_index.append([lookup[l], lookup[e]])               
    for d in os.listdir("Raw_data/data"):
        print(d)
        for f in glob.glob(os.path.join("Raw_data/data", d, "nowcast-clustering-16*")):
            graph = {"y": classes[d.split("_")[0]], "edge_index": edge_index, "x":[]}
            features = pd.read_csv(f, index_col=[0])
            features= features.replace('', -1, regex=True)
            # time_list
            for l in lookup:
                if l.startswith("create_dir_") or l.startswith("cleanup_"):
                    new_l = l.split("-")[0]
                else:
                    new_l = l
                job_features = features[features.index.str.startswith(new_l)][total_columns].values.tolist()[0]
                job_features = [-1 if x != x else x for x in job_features]   
                minim = min(job_features[2:8])
                job_features[2:8] = [ j-minim for j in job_features[2:8]]
                graph['x'].insert(lookup[l], job_features)     
            graphs.append(graph)
    return graphs
