# 1. Experiments details
## Install the required libraries
```bash
cd /dir
!pip install -r requirements.txt
```
## Fine-tune hyperparameters
```bash
%run fine_tune.py --dataset-name cifar10 --device cuda --epoch 50
```
## Run main.py for training
```bash
%run main.py --dataset-name (e.g. cifar10) --device cuda --epoch (e.g. 50) (other commands: e.g. --encoder-blocks-num 8 --heads-num 8)
```
## Run tensorborad for visualization
```bash
%load_ext tensorboard
%tensorboard --logdir ../checkpoints/model_folder
```
## Run main.py for testing
```bash
%run main.py --dataset-name cifar10 --device cuda --load-from ../checkpoints/model_folder/epoch_xx --load-model-config --epoch 10
```
## Ablation experiments
After removing or adding the corresponding layers, run main.py again to get the training results
```bash
%run main.py --dataset-name (e.g. cifar10) --device cuda --epoch (e.g. 50) (other commands: e.g. --encoder-blocks-num 8 --heads-num 8)
```
### Ablation experiments in MLP block
1. Removing Gelu layer in MLP block in file ..\vision_transformer\layers.py
```bash
    def forward(self, x):
        x = self.fc1(x)
        #x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
```
2. Adding Batchnorm layer in MLP block in file ..\vision_transformer\layers.py
```bash
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2) 
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2) 
```
### Ablation experiments in Encoder
1. Removing multi-head attention layer in Encoder in file ..\vision_transformer\modules.py
```bash
    def forward(self, x):
        #x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
```
2. Removing MLP layer in Encoder in file ..\vision_transformer\modules.py
```bash
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        #x = x + self.mlp(self.norm2(x))
```
3. Removing residuals layer in Encoder in file ..\vision_transformer\modules.py
```bash
    def forward(self, x):
        x =  self.attention(self.norm1(x))
        x =  self.mlp(self.norm2(x))
```
