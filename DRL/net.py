from torch import nn
import torch
import copy
import numpy as np

from torchinfo import summary






class ResNet_Conv2d(nn.Module):
    def __init__(self, input_dim, cells_per_block, layers_per_cell, latent_dim, dropout=0.0):
        super().__init__()

        self.cells_per_block = cells_per_block

        self.cells = nn.ModuleList()

        bottleneck_dim = int(latent_dim/4)

        self.input_layer = nn.Sequential(
            nn.Conv2d(input_dim, latent_dim, 1),
        )

        for _ in range(cells_per_block):
            self.cells.append( nn.Sequential(
                nn.BatchNorm2d(latent_dim),
                *([nn.Dropout2d(dropout)] if dropout>0 else []),
                nn.ReLU(),
                nn.Conv2d(latent_dim, bottleneck_dim, 1),
                *[l for _ in range(layers_per_cell) for l in [
                        nn.BatchNorm2d(bottleneck_dim),
                        *([nn.Dropout2d(dropout)] if dropout>0 else []),
                        nn.ReLU(),
                        nn.Conv2d(bottleneck_dim, bottleneck_dim, 3, padding=1),
                ] ],
                nn.BatchNorm2d(bottleneck_dim),
                *([nn.Dropout2d(dropout)] if dropout>0 else []),
                nn.ReLU(),
                nn.Conv2d(bottleneck_dim, latent_dim, 1),
            ) )

    def forward(self, input):
        x = self.input_layer(input)
        
        for i in range(self.cells_per_block):
            x = (x + self.cells[i](x))/2

        return x


class ResNet(nn.Module):
    def __init__(self, input_dim, cells_per_block, layers_per_cell, latent_dim, dropout=0.0):
        super().__init__()

        self.cells_per_block = cells_per_block

        self.cells = nn.ModuleList()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
        )

        for _ in range(cells_per_block):
            self.cells.append( nn.Sequential(
                nn.BatchNorm1d(latent_dim),
                *([nn.Dropout(dropout)] if dropout>0 else []),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim,),
                *[l for _ in range(layers_per_cell-1) for l in [
                        nn.BatchNorm1d(latent_dim),
                        *([nn.Dropout(dropout)] if dropout>0 else []),
                        nn.ReLU(),
                        nn.Linear(latent_dim, latent_dim),
                ] ]
            ) )


    def forward(self, input):
        x = self.input_layer(input)
        
        for i in range(self.cells_per_block):
            x = (x + self.cells[i](x))/2

        return x
        



class Net2048(nn.Module):
    def __init__(self, input_dim, action_dim, latent_dim, verbose=False, batch_size=1000):
        super().__init__()
        c, h, w = input_dim
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        dropout=0
        

        self.encoder = nn.Sequential(
            ResNet_Conv2d(input_dim[0], 2, 1, latent_dim[0], dropout=dropout),

            nn.Flatten(),
        ).float()

        self._latent_neurons = latent_dim[-2] * latent_dim[-1] * latent_dim[0]

        nodes_per_action = 64

        self.latent = nn.Sequential(

            nn.BatchNorm1d(self._latent_neurons),
            nn.ReLU(),
            
            ResNet(self._latent_neurons, 10, 1, action_dim*nodes_per_action, dropout=dropout),

            nn.BatchNorm1d(action_dim*nodes_per_action),
            *([nn.Dropout(dropout)] if dropout>0 else []),
            nn.ReLU(),

            nn.Linear(action_dim*nodes_per_action, action_dim*nodes_per_action),
            *([nn.Dropout(dropout)] if dropout>0 else []),
            nn.BatchNorm1d(action_dim*nodes_per_action),
            nn.ReLU(),

            nn.Linear(action_dim*nodes_per_action, action_dim),
            nn.PReLU(),
        ).float()

        self.train()


        if verbose:
            print("latent_dim", latent_dim)

            x1 = torch.zeros((batch_size,)+input_dim, dtype=torch.float)
            # x2 = torch.zeros((batch_size, info_dim), dtype=torch.float)
            y = self(x1)
            print('x1.shape, y.shape', x1.shape, y.shape)

            summary(self.encoder, input_size=(batch_size,)+input_dim, dtypes=[torch.float])
            summary(self.latent, input_size=(batch_size, self._latent_neurons), dtypes=[torch.float])
            
            summary(self, input_size=(batch_size,)+input_dim, dtypes=[torch.float, torch.float])



    def forward(self, input0):

        z = input0.to(torch.float)

        x = self.encoder(z)

        return self.latent(x)




class DeepQLearner(nn.Module):
    def __init__(self, input_dim, action_dim, latent_dim, batch_size=1000):
        super().__init__()

        self.best_historic = 0

        self.input_dim, self.action_dim, self.latent_dim = input_dim, action_dim, latent_dim

        net = Net2048(input_dim, action_dim, latent_dim, verbose=True, batch_size=batch_size)
        self.models = nn.ModuleDict({
            'online': net,
            'eval_model': copy.deepcopy(net),
        })

        self.sync_eval()

        for p in self.models['online'].parameters():
            p.requires_grad = True
        self.models['online'].train()
        


    def next_episode(self):
        pass


    
    def sync_eval(self):
        # assert self.initialized
        self.models["eval_model"].load_state_dict(self.models["online"].state_dict())
        self.models["eval_model"].eval()
        for p in self.models["eval_model"].parameters():
            p.requires_grad = False


    def sync_target(self, model):
        pass


    def forward(self, input, model):

        input = input.float()

        if model == 'eval_model':
            return self.models[model](input)

        elif model == 'online':
            return self.models[model](input)

        elif model == 'random':
            try:
                dev = input[0].get_device()
                # print(1)
                return torch.rand((len(input[0]), self.action_dim), device = f"cuda:{dev}", dtype=torch.float)
            except:
                # print(2)
                return torch.rand((len(input[0]), self.action_dim), device = f"cpu", dtype=torch.float)

        elif model == 'zeros':
            try:
                dev = input[0].get_device()
                # print(1)
                return torch.zeros((len(input[0]), self.action_dim), device = f"cuda:{dev}", dtype=torch.float)
            except:
                # print(2)
                return torch.zeros((len(input[0]), self.action_dim), device = f"cpu", dtype=torch.float)

        # elif model == 'target':
        #     x = self.historic_models[0](*input)
        #     for i in range(1, len(self.historic_models)):
        #         x += self.historic_models[i](*input)

        #     return x / len(self.historic_models)
            


