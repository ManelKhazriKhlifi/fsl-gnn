# fsl-gnn
This repository contains the code for the following approach named Few-shot Learning, Graph Neural Network, Contrastive Learning Remote Sensing imagery

## Brief Description
This repository is the approach designed to classify remote sensing imagery. This approach is a method that classifies remote sensing imagery using few-shot learning and a graph neural network. It has some models used for the classification mission.  
The advantage of this approach is to reduce the reliance on the label dataset.


## Requirement
torch>=1.8.0

torchvision>=0.9.0

torchaudio>=0.8.0

scikit-learn>=0.24.0

numpy>=1.19.0

tqdm>=4.60.0

Pillow>=8.0.0

matplotlib>=3.3.0

scipy>=1.6.0

pandas>=1.2.0

seaborn>=0.11.0


## Usage
clone this repository 
```ruby
https://github.com/0aub/fsl-gnn.git
```
## Dataset
the link of the datasheets are :

-UCMerced : http://weegee.vision.ucmerced.edu/datasets/landuse.html

-AID: https://pan.baidu.com/s/1mifOBv6#list/path=%2FA

-NWPU_RESISC45: https://www.kaggle.com/datasets/aqibrehmanpirzada/nwpuresisc45

-WHU-RS19: https://captain-whu.github.io/BED4RS/


## demo
You can train and test the whole process of RS image classification using this 

python3 train.py --dataset WHU-RS19-5-4-1 --num_shots 5 --batch_size 32 --dropout 0.1 --train 1 --test 1

python3 train.py --dataset WHU-RS19-8-1-1 --num_shots 1 --batch_size 32 --dropout 0.1 --train 1 --test 1

