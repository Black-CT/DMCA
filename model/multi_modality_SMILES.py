import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,global_mean_pool,GATConv,SAGEConv,Sequential,GraphNorm,global_max_pool,DeepGCNLayer
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN_branch_with_GAT(torch.nn.Module):
    def __init__(self):
        super(GNN_branch_with_GAT,self).__init__()
        self.graphconv=Sequential('x,edge_index,batch',[
            (GATConv(9, 128), 'x,edge_index -> x'),
            nn.LeakyReLU(),
            GraphNorm(128),
            (nn.Dropout(p=0.1), 'x -> x'),
            (GATConv(128, 256), 'x,edge_index -> x'),
            nn.LeakyReLU(),
            GraphNorm(256),
            (nn.Dropout(p=0.1), 'x -> x'),
            (GATConv(256, 512), 'x,edge_index -> x'),
            nn.LeakyReLU(),
            GraphNorm(512),
            (nn.Dropout(p=0.1), 'x -> x'),
            (GATConv(512, 767 * 2), 'x,edge_index -> x'),
            nn.LeakyReLU(),
            GraphNorm(767 * 2),

        ])


    def forward(self,data):
        x,edge_index,batch=data.x.float(), data.edge_index,data.batch
        graph_representation = self.graphconv(x,edge_index,batch)
        # print("graph_representation")
        # print(graph_representation)
        # print("------")
        mean=global_mean_pool(graph_representation,batch).view(-1,1,767*2)
        max=global_max_pool(graph_representation,batch).view(-1,1,767*2)
        graph_representation = torch.cat([mean,max], dim=1)
        return graph_representation

class Net(torch.nn.Module):
    def __init__(self, n_output_layers=1):
        super(Net,self).__init__()
        self.gnn_branch=GNN_branch_with_GAT()
        self.n_output_layers = n_output_layers
        self.model = AutoModelForMaskedLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

        self.outputlayer = nn.Sequential(
            nn.Linear(767, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 20),
            nn.LeakyReLU(),
            nn.Linear(20, self.n_output_layers),
            nn.Sigmoid(),
        )
        self.decodelayer1=nn.TransformerDecoderLayer(d_model=767, nhead=13,batch_first=True)
        # self.decodelayer2 = nn.TransformerDecoderLayer(d_model=767, nhead=13, batch_first=True)
    def forward(self,data,inputs):
        memory=self.gnn_branch(data)
        memory=memory.view(-1,4,767)
        tgt_mask=inputs.attention_mask.to(torch.float)
        tgt_key_padding_mask=tgt_mask
        tgt_key_padding_mask=torch.where(tgt_key_padding_mask == 0, 1.0, 0.0)
        # print(inputs.input_ids.shape)
        outputs = self.model(**inputs)
        tgt=outputs[0]
        # print("memory size:", memory.shape)
        # print("tgt size:", tgt.shape)
        # print("tgt")
        # print(tgt)
        # print("-----------")
        # print(memory)
        # print("------------")

        # key is smile embedding
        final_representation=self.decodelayer1(tgt,memory,tgt_key_padding_mask=tgt_key_padding_mask)
        # key is graph embedding
        # final_representation=self.decodelayer1(memory,tgt,memory_key_padding_mask=tgt_key_padding_mask)


        final_representationfinal_representation=final_representation[:,0,:]
        x=self.outputlayer(final_representationfinal_representation)
        return x

    def output_latent_space(self, data,inputs):
        memory=self.gnn_branch(data)
        # print("memory")
        # print(memory)
        # print("------------")

        memory=memory.view(-1,4,767)
        tgt_mask=inputs.attention_mask.to(torch.float)
        tgt_key_padding_mask=tgt_mask
        tgt_key_padding_mask=torch.where(tgt_key_padding_mask == 0, 1.0, 0.0)
        # print(inputs.input_ids.shape)
        outputs = self.model(**inputs)
        tgt=outputs[0]
        # print("memory size:", memory.shape)
        # print("tgt size:", tgt.shape)
        # print("tgt")
        # print(tgt)
        # print("-----------")
        # print(memory)
        # print("------------")

        final_representation=self.decodelayer1(memory, tgt,tgt_key_padding_mask=tgt_key_padding_mask)

        final_representationfinal_representation=final_representation[:,0,:]
        return final_representationfinal_representation


if __name__ == '__main__':
    x=torch.randn(3,9)
    edge_index=torch.tensor([[0,1,1,2],[1,0,2,1]],dtype=torch.long)
    drug = ["CC(C)[C@@H]1NC(=O)[C@H](C)OC(=O)C(NC(=O)[C@H](OC(=O)[C@@H](NC(=O)[C@H](C)O"
            "C(=O)[C@H](NC(=O)[C@H](OC(=O)[C@@H](NC(=O)[C@H](C)OC(=O)[C@H](NC(=O)[C@H](OC"
            "1=O)C(C)C)C(C)C)C(C)C)C(C)C)C(C)C)C(C)C)C(C)C)C(C)C"]
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    inputs = tokenizer(drug, padding=True, truncation=True, return_tensors="pt")


    model = Net()

    data1 = Data(x=x, edge_index=edge_index, id=1)
    data_list = [data1]
    loader = DataLoader(data_list,batch_size=1)
    for i, data in enumerate(loader):
        print(data)
        outputs = model.forward(data, inputs)
        print(outputs)

