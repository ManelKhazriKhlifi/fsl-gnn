import torch
from torch import nn
from torch.nn import functional as F



class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']
    def __init__(self, inplanes, planes, stride=1, 
                 downsample=None, groups=1, base_width=64, 
                 dilation=1):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, num_classes, layers=[1,1,1,1], groups=1, width_per_group=64):
        super(ResNet, self).__init__()
        block=Bottleneck
        self.num_classes = num_classes
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=False)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=False)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class GraphUnpool(nn.Module):
    def __init__(self):
        super(GraphUnpool, self).__init__()

    def forward(self, A, X, idx_batch):
        # optimized by Gai
        batch = X.shape[0]
        new_X = torch.zeros(batch, A.shape[1], X.shape[-1]).to(X.device)
        new_X[torch.arange(idx_batch.shape[0]).unsqueeze(-1), idx_batch] = X
        #
        return A, new_X

class GraphPool(nn.Module):
    def __init__(self, k, in_dim, num_classes, num_queries):
        super(GraphPool, self).__init__()
        self.k = k
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.proj = nn.Linear(in_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, A, X):
        batch = X.shape[0]
        idx_batch = []
        new_X_batch = []
        new_A_batch = []
        
        device = X.device
        # for each batch
        for i in range(batch):
            num_nodes = A[i, 0].shape[0]
            scores = self.proj(X[i])
            scores = torch.squeeze(scores)
            scores = self.sigmoid(scores/100)
            
            num_supports = num_nodes - self.num_queries
            support_scores = scores[:num_supports]
            intra_scores = support_scores - support_scores.mean()
            _, support_idx = torch.topk(intra_scores, int(self.k * num_supports), largest=False)
            support_values = support_scores[support_idx]
            query_values = scores[num_nodes - self.num_queries:]
            query_idx = torch.arange(num_nodes - self.num_queries, num_nodes).long().to(device)
            values = torch.cat([support_values, query_values], dim=0)
            idx = torch.cat([support_idx, query_idx], dim=0)

            new_X = X[i,idx, :]
            values = torch.unsqueeze(values, -1)
            new_X = torch.mul(new_X, values)
            new_A = A[i,idx, :]
            new_A = new_A[:, idx]
            idx_batch.append(idx)
            new_X_batch.append(new_X)
            new_A_batch.append(new_A)
        A = torch.stack(new_A_batch,dim=0).to(device)
        new_X = torch.stack(new_X_batch,dim=0).to(device)
        idx_batch = torch.stack(idx_batch,dim=0).to(device)
        return A, new_X, idx_batch

class MLP(nn.Module):
    def __init__(self,in_dim,hidden = 96, ratio=[2,2,1,1]):
        super(MLP, self).__init__()
        # set layers
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=hidden*ratio[0], kernel_size=1, bias=False),
                                    nn.BatchNorm2d(num_features=hidden*ratio[0]),
                                    nn.LeakyReLU())
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=hidden*ratio[0], out_channels=hidden*ratio[1], kernel_size=1, bias=False),
                                    nn.BatchNorm2d(num_features=hidden*ratio[1]),
                                    nn.LeakyReLU())
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=hidden * ratio[1], out_channels=hidden * ratio[2], kernel_size=1, bias=False),
                                    nn.BatchNorm2d(num_features=hidden * ratio[2]),
                                    nn.LeakyReLU())
        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels=hidden * ratio[2], out_channels=hidden * ratio[3], kernel_size=1, bias=False),
                                    nn.BatchNorm2d(num_features=hidden * ratio[3]),
                                    nn.LeakyReLU())
        self.conv_last = nn.Conv2d(in_channels=hidden * ratio[3], out_channels=1, kernel_size=1)

    def forward(self,X):
        # compute abs(x_i, x_j)
        x_i = X.unsqueeze(2)
        x_j = torch.transpose(x_i, 1, 2)
        x_ij = torch.abs(x_i - x_j)
        # parrallel
        x_ij = torch.transpose(x_ij, 1, 3).to(self.conv_last.weight.device)
        A_new = self.conv_last(self.conv_4(self.conv_3(self.conv_2(self.conv_1(x_ij))))).squeeze(1)

        A_new = F.softmax(A_new,dim=-1)

        return A_new

class GCN(nn.Module):
    def __init__(self, in_dim, out_dim=133,dropout=0.0):
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim*2, out_dim)
        self.drop = nn.Dropout(p=dropout)

    def forward(self,A_new, A_old, X):
        # parrallel
        X = X.to(self.proj.weight.device)
        A_new = A_new.to(X.device)
        A_old = A_old.to(X.device)
        #
        X = self.drop(X)
        #   
        X1 = torch.bmm(A_new, X)
        X2 = torch.bmm(A_old,X)
        X = torch.cat([X1,X2],dim=-1)
            
        X = self.proj(X)
        return X
    

class Unet(nn.Module):
    def __init__(self, pooling_ratios, in_dim, num_classes, num_queries, dropout_prob=0.1):
        super(Unet, self).__init__()
        l_n = len(pooling_ratios )
        self.l_n = l_n
        self.dropout_prob = dropout_prob
        
        # Start layers
        self.start_mlp = MLP(in_dim=in_dim)
        self.start_gcn = GCN(in_dim=in_dim, out_dim=in_dim)
        self.start_dropout = nn.Dropout(p=self.dropout_prob)

        # Down layers
        self.down_mlp = nn.ModuleList([MLP(in_dim=in_dim) for _ in range(l_n)])
        self.down_gcn = nn.ModuleList([GCN(in_dim=in_dim, out_dim=in_dim) for _ in range(l_n)])
        self.down_dropout = nn.ModuleList([nn.Dropout(p=self.dropout_prob) for _ in range(l_n)])
        self.pool = nn.ModuleList([GraphPool(pooling_ratios[l], in_dim=in_dim, num_classes=num_classes, num_queries=num_queries) for l in range(l_n)])
        self.unpool = nn.ModuleList([GraphUnpool() for _ in range(l_n)])
        
        # Bottom layers
        self.bottom_mlp = MLP(in_dim=in_dim)
        self.bottom_gcn = GCN(in_dim=in_dim, out_dim=in_dim)
        self.bottom_dropout = nn.Dropout(p=self.dropout_prob)

        # Up layers
        self.up_mlp = nn.ModuleList([MLP(in_dim=in_dim) for _ in range(l_n)])
        self.up_gcn = nn.ModuleList([GCN(in_dim=in_dim, out_dim=in_dim) for _ in range(l_n)])
        self.up_dropout = nn.ModuleList([nn.Dropout(p=self.dropout_prob) for _ in range(l_n)])

        # Output layers
        self.out_mlp = MLP(in_dim=in_dim * 2)
        self.out_gcn = GCN(in_dim=in_dim * 2, out_dim=num_classes)
        self.out_dropout = nn.Dropout(p=self.dropout_prob)

    def forward(self, A_init, X):
        adj_ms = []
        indices_list = []
        down_outs = []
        org_X = X
        A_old = A_init
        
        # Start layers
        A_new = self.start_mlp(X)
        X = self.start_gcn(A_new, A_old, X)
        X = self.start_dropout(X)
            
        # Downward pass
        for i in range(self.l_n):
            A_old = A_new
            A_new = self.down_mlp[i](X)
            X = self.down_gcn[i](A_new, A_old, X)
            X = self.down_dropout[i](X)
            adj_ms.append(A_new)
            down_outs.append(X)
            A_new, X, idx_batch = self.pool[i](A_new, X)
            indices_list.append(idx_batch)
                
        # Bottom layers
        A_old = A_new
        A_new = self.bottom_mlp(X)
        X = self.bottom_gcn(A_new, A_old, X)
        X = self.bottom_dropout(X)
            
        # Upward pass
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            A_old, idx_batch = adj_ms[up_idx], indices_list[up_idx]
            A_old, X = self.unpool[i](A_old, X, idx_batch)
            X = X.add(down_outs[up_idx])
            A_new = self.up_mlp[up_idx](X)
            X = self.up_gcn[up_idx](A_new, A_old, X)
            X = self.up_dropout[up_idx](X)
                
        # Output layers
        X = torch.cat([X, org_X], -1)
        A_old = A_new
        A_new = self.out_mlp(X)
        X = self.out_gcn(A_new, A_old, X)
        X = self.out_dropout(X)

        out = F.log_softmax(X, dim=-1)

        return out, X
