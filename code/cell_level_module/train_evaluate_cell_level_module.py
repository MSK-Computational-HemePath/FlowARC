import pandas as pd
import numpy as np
import os
import glob
import joblib
import fcsparser
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
import gc
from torchsampler import ImbalancedDatasetSampler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

#get train, test cases
path_normal_train, path_normal_test, path_abnormal_train, path_abnormal_test = joblib.load('train_test_labels.sav')

#create annotated dataset of normal and tumor/abnormal cells
train_normal = pd.DataFrame()
for case in path_normal_train:
    meta,tmp_df = fcsparser.parse(case, meta_data_only=False)
    tmp_df = pd.DataFrame(tmp_df)
    tmp_df = tmp_df.drop(columns=['Alexa Fluor 700-A', 'BV605-A','Time'], errors='ignore')
    tmp_df.columns=['FSC-A', 'FSC-H', 'SSC-A', 'SSC-H', 'CD20', 'CD34', 'CD10','CD33', 'CD58', 'CD45','CD19', 'CD38']
    tmp_df['case'] = os.path.basename(case)
    tmp_df['Dx'] = 0
    train_normal = pd.concat([train_normal,tmp_df])

train_abnormal = pd.DataFrame()
for case in path_abnormal_train:
    meta,tmp_df = fcsparser.parse(case, meta_data_only=False)
    tmp_df = pd.DataFrame(tmp_df)
    tmp_df = tmp_df.drop(columns=['Alexa Fluor 700-A', 'BV605-A','Time'], errors='ignore')
    tmp_df.columns=['FSC-A', 'FSC-H', 'SSC-A', 'SSC-H', 'CD20', 'CD34', 'CD10','CD33', 'CD58', 'CD45','CD19', 'CD38']
    tmp_df['case'] = os.path.basename(case)
    tmp_df['Dx'] = 1
    train_abnormal = pd.concat([train_abnormal,tmp_df])

test_normal = pd.DataFrame()
for case in path_normal_test:
    meta,tmp_df = fcsparser.parse(case, meta_data_only=False)
    tmp_df = pd.DataFrame(tmp_df)
    tmp_df = tmp_df.drop(columns=['Alexa Fluor 700-A', 'BV605-A','Time'], errors='ignore')
    tmp_df.columns=['FSC-A', 'FSC-H', 'SSC-A', 'SSC-H', 'CD20', 'CD34', 'CD10','CD33', 'CD58', 'CD45','CD19', 'CD38']
    tmp_df['case'] = os.path.basename(case)
    tmp_df['Dx'] = 0
    test_normal = pd.concat([test_normal,tmp_df])

test_abnormal = pd.DataFrame()
for case in path_abnormal_test:
    meta,tmp_df = fcsparser.parse(case, meta_data_only=False)
    tmp_df = pd.DataFrame(tmp_df)
    tmp_df = tmp_df.drop(columns=['Alexa Fluor 700-A', 'BV605-A','Time'], errors='ignore')
    tmp_df.columns=['FSC-A', 'FSC-H', 'SSC-A', 'SSC-H', 'CD20', 'CD34', 'CD10','CD33', 'CD58', 'CD45','CD19', 'CD38']
    tmp_df['case'] = os.path.basename(case)
    tmp_df['Dx'] = 1
    test_abnormal = pd.concat([test_abnormal,tmp_df])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed=1234

# define validation dataset
valid_normal = train_normal.sample(n=1000000, replace=False, random_state=seed, axis=0)
valid_abnormal = train_abnormal.sample(n=1000000, replace=False, random_state=seed, axis=0)

#merge normal and abnormal cells and shuffle
train_df = pd.concat([train_normal, train_abnormal])
train_df = train_df.sample(frac=1, replace=False, random_state=seed, axis=0).reset_index(drop=True)
test_df = pd.concat([test_normal, test_abnormal])
test_df = test_df.sample(frac=1, replace=False, random_state=seed, axis=0).reset_index(drop=True)
valid_df = pd.concat([valid_normal, valid_abnormal])
valid_df = valid_df.sample(frac=1, replace=False, random_state=seed, axis=0).reset_index(drop=True)

# define input and labels
train_x = train_df.drop(['case','Dx'],axis=1)
train_y = train_df['Dx']
test_x = test_df.drop(['case','Dx'],axis=1)
test_y = test_df['Dx']
valid_x = valid_df.drop(['case','Dx'],axis=1)
valid_y = valid_df['Dx']

#split train dataset due to limitations of imbalanced sampler
train_x1 = train_x[:14000000]
train_x2 = train_x[14000000:]
train_y1 = train_y[:14000000]
train_y2 = train_y[14000000:]


train_X1 = torch.unsqueeze(torch.tensor(np.float32(train_x1)),1)
train_X2 = torch.unsqueeze(torch.tensor(np.float32(train_x2)),1)

test_X = torch.unsqueeze(torch.tensor(np.float32(test_x)),1)
valid_X = torch.unsqueeze(torch.tensor(np.float32(valid_x)),1)

train_Y1 = torch.tensor(np.float32(train_y1))
train_Y2 = torch.tensor(np.float32(train_y2))
test_Y = torch.tensor(np.float32(test_y))
valid_Y = torch.tensor(np.float32(valid_y))


train_dataset1 = TensorDataset(train_X1,train_Y1)
train_dataset2 = TensorDataset(train_X2,train_Y2)
test_dataset = TensorDataset(test_X,test_Y)
valid_dataset = TensorDataset(valid_X,valid_Y)

num_epochs = 100
batch_size = 800
learning_rate = 0.1

train_loader1 = DataLoader(train_dataset1, batch_size=batch_size, shuffle=False,num_workers=24,pin_memory=True, sampler=ImbalancedDatasetSampler(train_dataset1))
train_loader2 = DataLoader(train_dataset2, batch_size=batch_size, shuffle=False,num_workers=24,pin_memory=True, sampler=ImbalancedDatasetSampler(train_dataset2))
test_loader = DataLoader(test_dataset, batch_size=batch_size*30, shuffle=False,num_workers=24,pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size*30, shuffle=False,num_workers=24,pin_memory=True)

# 1D ResNet model
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
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

    def __init__(self, block, layers, in_features=1, num_outputs=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv1d(in_features, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
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
        out = x
        x = self.fc(x)

        return x, out

    def forward(self, x):
        return self._forward_impl(x)

# OP Loss
class OrthogonalProjectionLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(OrthogonalProjectionLoss, self).__init__()
        self.gamma = gamma

    def forward(self, features, labels=None):
        #  features are normalized
        features = F.normalize(features, p=2, dim=1)

        labels = labels[:, None]  # extend dim

        mask = torch.eq(labels, labels.t()).bool().to(device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)

        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = (mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)  # TODO: removed abs

        loss = (1.0 - pos_pairs_mean) + self.gamma * neg_pairs_mean

        return loss

#Define model
model = ResNet(BasicBlock, [2, 2, 2, 2], num_outputs=2)
model.to(device)


optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum = 0.9)
op_loss = OrthogonalProjectionLoss(gamma=0.5)
loss_func = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10,20,30,50,70], gamma=0.1)

epoch_iter=1
for _ in tqdm(range(num_epochs), desc="epoch", leave=False):
    # train
    model.train()
    prog_iter1 = tqdm(train_loader1, desc="Training1", leave=False)
    prog_iter2 = tqdm(train_loader2, desc="Training2", leave=False)
    all_pred_probt = []
    trlab=[]
    tslab=[]
    train_loss=0
    test_loss=0
    batch_idx1 = 0
    batch_idx2 = 0 
    for batch_idx, batch in enumerate(prog_iter1):
        
        input_x, input_y = tuple(t for t in batch)
        input_y = input_y.type(torch.LongTensor)
        input_x, input_y = input_x.to(device), input_y.to(device)
        
        optimizer.zero_grad()
        pred = model(input_x)
        alpha = 1
        loss =  loss_func(pred[0], input_y) + alpha*op_loss(pred[1], input_y)         
        loss.backward()
        optimizer.step()
        
    for batch_idx, batch in enumerate(prog_iter2):
        
        input_x, input_y = tuple(t for t in batch)
        input_y = input_y.type(torch.LongTensor)
        input_x, input_y = input_x.to(device), input_y.to(device)
        optimizer.zero_grad()
        pred = model(input_x)
        alpha = 1
        loss =  loss_func(pred[0], input_y) + alpha*op_loss(pred[1], input_y)         
        loss.backward()
        optimizer.step() 
    scheduler.step()
    
    #Evaluate test and validation at specific intervals
    if epoch_iter%5==0:
        model.eval()
        prog_iter_valid = tqdm(valid_loader, desc="Validation", leave=False)
        prog_iter_test = tqdm(test_loader, desc="Testing", leave=False)
        with torch.no_grad():
            for batch_idx, batch in enumerate(prog_iter_valid):

                input_x, input_y = tuple(t for t in batch)
                trlab.append(input_y.tolist())
                input_y = input_y.type(torch.LongTensor)
                input_x, input_y = input_x.to(device), input_y.to(device)

                pred = model(input_x)
                all_pred_probt.append(pred[0].cpu().data.numpy())


        all_pred_probt = np.concatenate(all_pred_probt)
        trlab = np.concatenate(trlab)
        all_predt = np.argmax(all_pred_probt, axis=1)
        print(classification_report(trlab, all_predt))
        all_pred_prob = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(prog_iter_test):
                input_x, input_y = tuple(t for t in batch)
                tslab.append(input_y.tolist())
                input_y = input_y.type(torch.LongTensor)
                input_x, input_y = input_x.to(device), input_y.to(device)
                pred = model(input_x)
                loss =  loss_func(pred[0], input_y) + alpha*op_loss(pred[1], input_y)
                test_loss += loss.item()
                all_pred_prob.append(pred[0].cpu().data.numpy())
                batch_idx2 = batch_idx
        
        avg_test_loss = test_loss / (batch_idx2+1)
        all_pred_prob = np.concatenate(all_pred_prob)
        tslab = np.concatenate(tslab)
        all_pred = np.argmax(all_pred_prob, axis=1)
        print(classification_report(tslab, all_pred))
    epoch_iter+=1

torch.save(model.state_dict(), '../model/cell_level_module_auto')