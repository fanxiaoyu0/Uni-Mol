import numpy as np
import pandas as pd
import pickle as pkl
import random
import os
import pdb
from tqdm import tqdm
from threading import Thread, Lock
from rdkit import Chem
from rdkit.Chem import AllChem
from deepchem import molnet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from unicore.modules import init_bert_params
from unicore.data import (
    Dictionary, NestedDictionaryDataset, TokenizeDataset, PrependTokenDataset,
    AppendTokenDataset, FromNumpyDataset, RightPadDataset, RightPadDataset2D,
    RawArrayDataset, RawLabelDataset,
)
from unimol.data import (
    KeyDataset, ConformerSampleDataset, AtomTypeDataset,
    RemoveHydrogenDataset, CroppingDataset, NormalizeDataset,
    DistanceDataset, EdgeTypeDataset, RightPadDatasetCoord, 
)
from unimol.models.transformer_encoder_with_pair import TransformerEncoderWithPair
from unimol.models.unimol import NonLinearHead, GaussianLayer

def set_random_seed(random_seed=1024):
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

class UniMolModel(nn.Module):
    def __init__(self):
        super().__init__()
        dictionary = Dictionary.load('../data/raw/token_list.txt')
        dictionary.add_symbol("[MASK]", is_special=True)
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(
            len(dictionary), 512, self.padding_idx
        )
        self._num_updates = None
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=15,
            embed_dim=512,
            ffn_embed_dim=2048,
            attention_heads=64,
            emb_dropout=0.1,
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.0,
            max_seq_len=512,
            activation_fn='gelu',
            no_final_head_layer_norm=True,
        )

        K = 128
        n_edge_type = len(dictionary) * len(dictionary)
        self.gbf_proj = NonLinearHead(
            K, 64, 'gelu'
        )
        self.gbf = GaussianLayer(K, n_edge_type)

        self.apply(init_bert_params)

    def forward(
        self,
        sample,
    ):
        input = sample['input']
        src_tokens, src_distance, src_coord, src_edge_type \
            = input['src_tokens'], input['src_distance'], input['src_coord'], input['src_edge_type']
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(src_tokens)

        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(src_distance, src_edge_type)
        (encoder_rep, encoder_pair_rep, delta_encoder_pair_rep, x_norm, delta_encoder_pair_rep_norm) \
            = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        output = {
            "molecule_representation": encoder_rep[:, 0, :],  # get cls token
            "smiles": sample['input']["smiles"],
        }
        return output

def get_dataset_from_deepchem():
    tasks, datasets, transformers = molnet.load_lipo(splitter='scaffold')
    train_dataset, validate_dataset, test_dataset = datasets
    dataset = {
        "train": train_dataset,
        "validate": validate_dataset,
        "test": test_dataset,
    }
    smiles_list = []
    label_list = []
    dataset_type_list = []
    for dataset_type, dataset in dataset.items():
        smiles_list.extend(dataset.ids.tolist())
        label_list.extend(dataset.y.squeeze().tolist())
        dataset_type_list.extend([dataset_type] * len(dataset.ids))
    data_df = pd.DataFrame({
            "smiles": smiles_list,
            "label": label_list,
            "dataset_type": dataset_type_list,
        }
    )
    data_df.to_csv("../data/raw/data_df.csv", index=False)

def calculate_3D_structure():
    def get_smiles_list_():
        data_df = pd.read_csv("../data/raw/data_df.csv")
        smiles_list = data_df["smiles"].tolist()
        smiles_list = list(set(smiles_list))
        print(len(smiles_list))
        return smiles_list

    def calculate_3D_structure_(smiles_list):
        n = len(smiles_list)
        global p
        index = 0
        while True:
            mutex.acquire()
            if p >= n:
                mutex.release()
                break
            index = p
            p += 1
            mutex.release()

            smiles = smiles_list[index]
            print(index, ':', round(index / n * 100, 2), '%', smiles)

            molecule = Chem.MolFromSmiles(smiles)
            molecule = AllChem.AddHs(molecule)
            atoms = [atom.GetSymbol() for atom in molecule.GetAtoms()]
            coordinate_list = []
            result = AllChem.EmbedMolecule(molecule, randomSeed=42)
            if result != 0:
                print('EmbedMolecule failed', result, smiles)
                mutex.acquire()
                with open('../data/result/invalid_smiles.txt', 'a') as f:
                    f.write('EmbedMolecule failed' + ' ' + str(result) + ' ' + str(smiles) + '\n')
                mutex.release()
                continue
            try:
                AllChem.MMFFOptimizeMolecule(molecule)
            except:
                print('MMFFOptimizeMolecule error', smiles)
                mutex.acquire()
                with open('../data/result/invalid_smiles.txt', 'a') as f:
                    f.write('MMFFOptimizeMolecule error' + ' ' + str(smiles) + '\n')
                mutex.release()
                continue
            coordinates = molecule.GetConformer().GetPositions()
            
            assert len(atoms) == len(coordinates), "coordinates shape is not align with {}".format(smiles)
            coordinate_list.append(coordinates.astype(np.float32))

            global smiles_to_conformation_dict
            mutex.acquire()
            smiles_to_conformation_dict[smiles] = {'smiles': smiles, 'atoms': atoms, 'coordinates': coordinate_list}
            mutex.release()  

    mutex = Lock()
    smiles_list = get_smiles_list_()
    global smiles_to_conformation_dict
    smiles_to_conformation_dict = {}
    global p
    p = 0
    thread_count = 16
    threads = []
    for i in range(thread_count):
        threads.append(Thread(target=calculate_3D_structure_, args=(smiles_list, )))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    pkl.dump(smiles_to_conformation_dict, open('../data/intermediate/smiles_to_conformation_dict.pkl', 'wb'))
    print('Valid smiles count:', len(smiles_to_conformation_dict))

def construct_data_list():
    data_df = pd.read_csv("../data/raw/data_df.csv")
    smiles_to_conformation_dict = pkl.load(open('../data/intermediate/smiles_to_conformation_dict.pkl', 'rb'))
    data_list = []
    for index, row in data_df.iterrows():
        smiles = row["smiles"]
        if smiles in smiles_to_conformation_dict:
            data_item = {
                "atoms": smiles_to_conformation_dict[smiles]["atoms"],
                "coordinates": smiles_to_conformation_dict[smiles]["coordinates"],
                "smiles": smiles,
                "label": row["label"],
                "dataset_type": row["dataset_type"],
            }
            data_list.append(data_item)
    pkl.dump(data_list, open('../data/intermediate/data_list.pkl', 'wb'))

def convert_data_list_to_data_loader(remove_hydrogen):
    def convert_data_list_to_dataset_(data_list):
        dictionary = Dictionary.load('../data/raw/token_list.txt')
        dictionary.add_symbol("[MASK]", is_special=True)
        smiles_dataset = KeyDataset(data_list, "smiles")
        label_dataset = KeyDataset(data_list, "label")
        dataset = ConformerSampleDataset(data_list, 1024, "atoms", "coordinates")
        dataset = AtomTypeDataset(data_list, dataset)
        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", remove_hydrogen, False)
        dataset = CroppingDataset(dataset, 1, "atoms", "coordinates", 256)
        dataset = NormalizeDataset(dataset, "coordinates", normalize_coord=True)
        token_dataset = KeyDataset(dataset, "atoms")
        token_dataset = TokenizeDataset(token_dataset, dictionary, max_seq_len=512)
        coord_dataset = KeyDataset(dataset, "coordinates")
        src_dataset = AppendTokenDataset(PrependTokenDataset(token_dataset, dictionary.bos()), dictionary.eos())
        edge_type = EdgeTypeDataset(src_dataset, len(dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        coord_dataset = AppendTokenDataset(PrependTokenDataset(coord_dataset, 0.0), 0.0)
        distance_dataset = DistanceDataset(coord_dataset)
        return NestedDictionaryDataset({
            "input": {
                "src_tokens": RightPadDataset(src_dataset, pad_idx=dictionary.pad(),),
                "src_coord": RightPadDatasetCoord(coord_dataset, pad_idx=0,),
                "src_distance": RightPadDataset2D(distance_dataset, pad_idx=0,),
                "src_edge_type": RightPadDataset2D(edge_type, pad_idx=0,),
                "smiles": RawArrayDataset(smiles_dataset),
            }, 
            "target": {
                "label": RawLabelDataset(label_dataset),
            }
        })

    batch_size = 64
    data_list = pkl.load(open('../data/intermediate/data_list.pkl', 'rb'))
    data_list_train = [data_item for data_item in data_list if data_item["dataset_type"] == "train"]
    data_list_validate = [data_item for data_item in data_list if data_item["dataset_type"] == "validate"]
    data_list_test = [data_item for data_item in data_list if data_item["dataset_type"] == "test"] 
    dataset_train = convert_data_list_to_dataset_(data_list_train)
    dataset_validate = convert_data_list_to_dataset_(data_list_validate)
    dataset_test = convert_data_list_to_dataset_(data_list_test)
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=dataset_train.collater)
    data_loader_valid = DataLoader(dataset_validate, batch_size=batch_size, shuffle=True, collate_fn=dataset_validate.collater)
    data_loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, collate_fn=dataset_test.collater)
    return data_loader_train, data_loader_valid, data_loader_test

class UniMolRegressor(nn.Module):
    def __init__(self, remove_hydrogen):
        super().__init__()
        self.encoder = UniMolModel()
        if remove_hydrogen:
            self.encoder.load_state_dict(torch.load('../weight/mol_pre_no_h_220816.pt')['model'], strict=False)
        else:
            self.encoder.load_state_dict(torch.load('../weight/mol_pre_all_h_220816.pt')['model'], strict=False)
        self.mlp = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def move_batch_to_cuda(self, batch):
        batch['input'] = { k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch['input'].items() }
        batch['target'] = { k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch['target'].items() }
        return batch

    def forward(self, batch):
        batch = self.move_batch_to_cuda(batch)
        encoder_output = self.encoder(batch)
        molecule_representation = encoder_output['molecule_representation']
        smiles_list = encoder_output['smiles']
        x = self.mlp(molecule_representation)
        return x

def evaluate(model, data_loader):
    model.eval()
    label_predict = torch.tensor([], dtype=torch.float32).cuda()
    label_true = torch.tensor([], dtype=torch.float32).cuda()
    with torch.no_grad():
        # for batch in data_loader:
        for batch in tqdm(data_loader):
            label_predict_batch = model(batch)
            label_true_batch = batch['target']['label']

            label_predict = torch.cat((label_predict, label_predict_batch.detach()), dim=0)
            label_true = torch.cat((label_true, label_true_batch.detach()), dim=0)
    
    label_predict = label_predict.cpu().numpy()
    label_true = label_true.cpu().numpy()
    rmse = round(np.sqrt(mean_squared_error(label_true, label_predict)), 3)
    mae = round(mean_absolute_error(label_true, label_predict), 3)
    r2 = round(r2_score(label_true, label_predict), 3)
    metric = {'rmse': rmse, 'mae': mae, 'r2': r2}
    return metric

def train(model_version, remove_hydrogen=True):
    data_loader_train, data_loader_validate, data_loader_test = convert_data_list_to_data_loader(remove_hydrogen)

    model = UniMolRegressor(remove_hydrogen)
    model.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 15)

    current_best_metric = 1e10
    max_bearable_epoch = 50
    current_best_epoch = 0
    for epoch in range(300):
        model.train()
        # for batch in data_loader_train:
        for batch in tqdm(data_loader_train):
            label_predict_batch = model(batch)
            label_true_batch = batch['target']['label'].unsqueeze(1).to(torch.float32)
            # print(label_predict_batch.shape, label_true_batch.shape)
            # print(label_predict_batch)
            # print(label_true_batch)
            # pdb.set_trace()
            
            loss = criterion(label_predict_batch, label_true_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        metric_train = evaluate(model, data_loader_train)
        metric_validate = evaluate(model, data_loader_validate)
        metric_test = evaluate(model, data_loader_test)

        if metric_validate['rmse'] < current_best_metric:
            current_best_metric = metric_validate['rmse']
            current_best_epoch = epoch
            torch.save(model.state_dict(), "../weight/" + model_version + ".pt")

        print("==================================================================================")
        print('Epoch', epoch)
        print('Train', metric_train)
        print('validate', metric_validate)
        print('Test', metric_test)
        print('current_best_epoch', current_best_epoch, 'current_best_metric', current_best_metric)
        print("==================================================================================")
        if epoch > current_best_epoch + max_bearable_epoch:
            break

def test(model_version, remove_hydrogen=True):
    data_loader_train, data_loader_validate, data_loader_test = convert_data_list_to_data_loader(remove_hydrogen)

    model = UniMolRegressor(remove_hydrogen)
    model.load_state_dict(torch.load("../weight/" + model_version + ".pt"))
    model.cuda()

    metric_train = evaluate(model, data_loader_train)
    metric_validate = evaluate(model, data_loader_validate)
    metric_test = evaluate(model, data_loader_test)
    print("Train", metric_train)
    print("validate", metric_validate)
    print("Test", metric_test)

if __name__ == "__main__":
    set_random_seed(1024)
    # get_dataset_from_deepchem()
    # calculate_3D_structure()
    # construct_data_list()
    train(model_version='0', remove_hydrogen=True)
    # test(model_version='0', remove_hydrogen=True)
    print('All is well!')
