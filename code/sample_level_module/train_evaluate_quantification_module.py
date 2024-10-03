import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.utils as sku
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, RocCurveDisplay, auc,r2_score,mean_squared_error,mean_absolute_error
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset, DataLoader, Dataset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import gc
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 50
batch_size = 10
learning_rate = 5e-2

def calculate_2dft(input):
    ft = np.fft.ifftshift(input)
    ft = np.fft.fft2(ft)
    ft = np.fft.fftshift(ft)
    return [input,ft.real,ft.imag]

labels_train, fraction_train, accuracy_train = joblib.load('../tmp/Training_input_auto.sav')
labels_test, fraction_test, accuracy_test = joblib.load('../tmp/Testing_input_auto.sav')

class TorchGenerator(Dataset):
    # Constructor
    def __init__(self, path, label_array):
        self.y = label_array
        self.path = path
    def __len__(self):
        return round(0.5*len(self.y))

    # Getter
    def __getitem__(self, idx):
        tmp_df = pd.read_pickle(self.path+"df_"+str(round(0.5*len(self.y))+idx+1))
        tot = tmp_df.shape[0]
        tots = np.float32(tot/100000)
        num=7500
        tmp_x = tmp_df[:num]
        dxsum = tmp_x['Dx'].sum()
        tmp_x = tmp_x.drop(['case','Dx','pred'],axis=1)
        tmp_x = calculate_2dft(tmp_x)
        tmp_x[0].iat[0, 0] =tots
        tmp_mega = tmp_x
        samplex = torch.tensor(np.float32(tmp_mega))
        if dxsum<num:
            sampley = torch.tensor(np.float32(dxsum/num))
        else:
            sampley = torch.tensor(np.float32(1))
        return samplex, sampley
    
train_dataset = TorchGenerator( "/global_data/projects/flow/training_input_auto/", labels_train)
test_dataset = TorchGenerator( "/global_data/projects/flow/testing_input_auto/", labels_test)


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
        features = x
        x = self.fc(x)

        return x , features

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


def train(model, device, train_loader, optimizer, epoch):
    # Set the model to training mode
    model.train()
    train_loss = 0
    # Process the images in batches
    for batch_idx, (data, target) in enumerate(train_loader):
        target = torch.unsqueeze(target,1)
        data, target = data.to(device), target.to(device)
        # Reset the optimizer
        optimizer.zero_grad()
        # Push the data forward through the model layers
        output = model(data)
        # Get the loss
        loss = loss_criteria(output[0], target)
        # Keep a running total
        train_loss += loss.item()
        # Backpropagate
        loss.backward()
        optimizer.step()

    avg_loss = train_loss / (batch_idx+1)
    
    scheduler.step()

    
    return avg_loss
            
    
    
def compute_metrics(model, test_loader):
    
    model.eval()
    
    val_loss = 0
    val_correct = 0
        
    score_list   = torch.Tensor([]).to(device)
    pred_list    = torch.Tensor([]).to(device)
    target_list  = torch.Tensor([]).to(device)
    
    
    for iter_num, (image, target) in enumerate(test_loader):
        
        target = torch.unsqueeze(target,1)
        image, target = image.to(device), target.to(device)
        # Compute the loss
        with torch.no_grad():
            output = model(image)
        # Log loss
        val_loss += loss_criteria(output[0], target).item()
 
        # Calculate the number of correctly classified examples
        pred = output[0]
        
        # Bookkeeping 
        pred_list    = torch.cat([pred_list, pred.squeeze()])
        target_list  = torch.cat([target_list, target.squeeze()])

    mae = mean_absolute_error(target_list.tolist(), pred_list.tolist())
    mse = mean_squared_error(target_list.tolist(), pred_list.tolist())
    r2 = r2_score(target_list.tolist(), pred_list.tolist())
    
    # put together values
    metrics_dict = {"Mean absolute Error": mae,
                    "Mean squared Error": mse,
                    "R2 Score": r2,
                    "pred_list": pred_list.tolist(),
                    "target_list": target_list.tolist()}
    
    
    return metrics_dict

r2_array=[]
mae_array=[]
mse_array=[]
pred_array=[]
label_array=[]
for iteration in range(5):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12,pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=12,pin_memory=True)
    
    model = ResNet101(img_channel=3, num_classes=1)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_criteria = nn.HuberLoss(delta=0.007)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10,15,20,30], gamma=0.1)
    
    epoch_nums = []
    epochs = num_epochs
    for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch)
    metrics_dict2 = compute_metrics(model, test_loader)
    r2_array.append(metrics_dict2["R2 Score"])
    mae_array.append(metrics_dict2["Mean absolute Error"])
    mse_array.append(metrics_dict2["Mean squared Error"])
    pred_array.append(metrics_dict2["pred_list"])
    torch.save(model.state_dict(), '../model/quantification_module_auto_bootstrap_'+str(iteration+1))
    print('Completed Model '+str(iteration+1))
    print('R2 Score: '+str(metrics_dict2["R2 Score"]))

print('R-squared with 95% CI'+str(np.percentile(r2_array, (2.5, 97.5))))
print('Mean Absolute Error with 95% CI'+str(np.percentile(mae_array, (2.5, 97.5))))
print('Mean Squared Error with 95% CI'+str(np.percentile(mse_array, (2.5, 97.5))))