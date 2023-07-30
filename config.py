import os

class Config():
    curPath = os.path.abspath(__file__)
    rootPath = os.path.split(curPath)[0]


    DVS128_root_dir = os.path.join(rootPath, 'data/DVS128-Gait/dual_graph')
    Har_root_dir = os.path.join(rootPath, 'data/hardvs')
    NMinist_root_dir = os.path.join(rootPath, 'data/N_Mnist')
    ASL_root_dir = os.path.join(rootPath, 'data/ASL')
    
    log_dir = os.path.join(rootPath, 'log_dir')
    graph_train_log_path = os.path.join(log_dir, 'dvs128_dual_0420_512_3.log')
    model_dir = os.path.join(rootPath, 'trained_model')
    gcn_model_name = os.path.join(model_dir, 'DVS128_train_{}.pkl')
