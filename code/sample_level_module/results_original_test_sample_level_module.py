import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.utils as sku
import fcsparser
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset, DataLoader, Dataset
import gc
import joblib
import seaborn as sns
from tqdm import tqdm

device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

labels_test_abnormal = pd.read_csv('labels_test_abnormal.csv',sep=",",header=None)
labels_test_normal = pd.read_csv('labels_test_normal.csv',sep=",",header=None)
labels = np.append(labels_test_normal[1],labels_test_abnormal[1])
labels_test = np.append(labels_test_normal[0],labels_test_abnormal[0])

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
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)  

model1 = ResNet(BasicBlock, [2, 2, 2, 2], num_outputs=2)
model1.to(device1)
model1.load_state_dict(torch.load('../model/cell_level_module_auto', map_location=device1))
model1.eval()

def get_prediction( model, dataframe):
    
    tmp_df = dataframe
    test_x = tmp_df
    test_y = np.ones(tmp_df.shape[0])
    test_X = torch.unsqueeze(torch.tensor(np.float32(test_x)),1)
    test_Y = torch.tensor(np.float32(test_y))
    test_dataset = TensorDataset(test_X,test_Y)
    test_loader = DataLoader(test_dataset, batch_size=80000, shuffle=False)    
    prog_iter_test = tqdm(test_loader, desc="Testing", leave=False)
    all_pred_prob = []
    with torch.no_grad():
        
        for batch_idx, (input_x, input_y) in enumerate(prog_iter_test):

            input_y = input_y.type(torch.LongTensor)
            input_x, input_y = input_x.to(device1), input_y.to(device1)
            out = model(input_x)
            pred = F.softmax(out, dim=1)
            all_pred_prob.append(pred.cpu().data.numpy())
    
    all_pred_prob = np.concatenate(all_pred_prob)
    all_pred = np.argmax(all_pred_prob, axis=1)
    return all_pred_prob, all_pred

nchoose=7500

def add_cellID(model, tmp_mega):
    pred_te = get_top_50( model, tmp_mega)
    tmp_mega['pred'] = pred_te[1]
    tmp_mega['pred_0'] = pred_te[0][:,0]
    tmp_mega['pred_1'] = pred_te[0][:,1]
    tmp_mega = tmp_mega.sort_values(['pred_0', 'pred_1'], ascending=[True, False])
    if tmp_mega.shape[0]>nchoose:
        return tmp_mega[:nchoose]
    else:
        return tmp_mega
    
test_folder = '../tmp/cases_b_cells/'


class block(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super().__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels,
            intermediate_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            image_channels, 64, kernel_size=5, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

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
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)

def ResNet101(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 23, 3], img_channel, num_classes)

model = ResNet101(img_channel=3, num_classes=2)
model.to(device)
swa_model = model
swa_model.load_state_dict(torch.load('../model/sample_level_module_auto', map_location=device))
swa_model.eval()

def calculate_2dft(input):
    ft = np.fft.ifftshift(input)
    ft = np.fft.fft2(ft)
    ft = np.fft.fftshift(ft)
    return [input,ft.real,ft.imag]

prediction=[]
prediction_prob=[]
labell = []
label_idx = []
for idx in range(0,len(labels_test)):
    folder = test_folder
    meta,tmp_df = fcsparser.parse(folder+labels_test[idx], meta_data_only=False)
    tmp_df = pd.DataFrame(tmp_df)
    tmp_df = tmp_df.drop(columns=['Alexa Fluor 700-A', 'BV605-A','Time'], errors='ignore')
    if len(tmp_df.columns)==12:
        tmp_df.columns=['FSC-A', 'FSC-H', 'SSC-A', 'SSC-H', 'CD20', 'CD34', 'CD10','CD33', 'CD58', 'CD45','CD19', 'CD38']
    else:
        continue
    tmp_df = add_cellID(model1, tmp_df)
    if tmp_df.shape[0]==nchoose:
        
        tmp_x = tmp_df.drop(['pred'],axis=1)
        tmp_x = calculate_2dft(tmp_x)
        tmp_mega = tmp_x
        samplex = torch.unsqueeze(torch.tensor(np.float32(tmp_mega)),0)
        output = swa_model(samplex.to(device))
        pred_prob = F.softmax(output.data, dim=1)[0]
        _, pred = torch.max(output.data, 1)
        prediction.append(pred.cpu().numpy()[0])
        prediction_prob.append(pred_prob.cpu().numpy()[1])
        labell.append(labels[idx])
        label_idx.append(idx)
        #if pred.cpu().numpy()[0]!= labels[idx]:
        #    print(labels_test[idx],idx,pred_prob.cpu().numpy()[1])
    gc.collect()
    torch.cuda.empty_cache()

classification_metrics = classification_report(labell, prediction,target_names = ['Healthy Sample', 'Diseased Sample'],output_dict= True)
sensitivity = classification_metrics['Diseased Sample']['recall']
specificity = classification_metrics['Healthy Sample']['recall']
accuracy = classification_metrics['accuracy']
print('------------------- Test Performance --------------------------------------')
print("Accuracy \t {:.3f}".format(accuracy))
print("Sensitivity \t {:.3f}".format(sensitivity))
print("Specificity \t {:.3f}".format(specificity))
print("------------------------------------------------------------------------------")

conf_matrix = confusion_matrix(labell, prediction)
#annot = [["225\n\n (46.9%)","6\n\n (1.3%)"],["4\n\n (0.8%)","245\n\n (51.0%)"]]
ax= plt.subplot()
sns.heatmap(conf_matrix, annot=True, ax = ax, cmap = 'Greens', fmt="", annot_kws={'size': 14});
ax.set_xlabel('Predicted labels',fontsize=14);ax.set_ylabel('True labels',fontsize=14); 
ax.set_title('Confusion Matrix',fontsize=15); 
ax.xaxis.set_ticklabels(['Normal','Tumor'],fontsize=13); ax.yaxis.set_ticklabels(['Normal','Tumor'],fontsize=13)
plt.savefig('Big_Real_confmat.png', dpi=500);