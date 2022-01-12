# Networks to classify Tsinghua Dogs Dataset 
This repository contains convolution neural networks that classfies species of dogs using Tsinghua Dogs Dataset. The dataset is provided in [this link](https://cg.cs.tsinghua.edu.cn/ThuDogs/). 
# Requirements
**Step1: Install Requirements**
To install all the packages required to run the codes, enter the following to the terminal. 
```shell
git clone https://github.com/georgeNakayama/ThuDogs.git 
cd ThuDogs
python -m pip install -r requirements.txt
```
If you have any installation problems for Jittor, please refer to [Jittor](https://github.com/Jittor/jittor)

**Step2: Setup Environment Variable**
Now we need to setup enviroment variable to run the python scripts. Add 
```shell
export PYTHONPATH=$PYTHONPATH:{you_own_path}/ThuDogs
```
to ```.bashrc``` or ```.zshrc``` depending on the shell you use. 
Then, run 
```shell
source .bashrc
```
or 
```shell
source .zshrc
```
respectively. You are good to go. 

# Getting Started
ThuDogs runs all of its training and testing through ```config-file```. Please refer to [config.md](docs/config.md) for details. 

# Dataset
Download and unzip the Tsinghua Dogs Dataset by running from ThuDogs directory 
```shell
python tools/process.py --download --zip --save_dir {destination to save the dataset}
```
To use the dataset, we need to change the directory name for the images to **images**, the annotations to **annotations** and the train/validation split list directory to **datalist**. 

# Train
We now support the training of two types of networks. One of them is [Resnet50](https://arxiv.org/pdf/1512.03385.pdf). To train this network we can run 
```shell
python tools/main.py --config-file config/rnet50-sgd-consine-cusdataset.py --task=train
```
To resume a training using checkpoints, add ```resume_path={you_checkpointspath}``` to the last line of the config file. 

The second network we support is [PMG](https://arxiv.org/pdf/2003.03836.pdf) where the training can be run by typing 
```shell
python tools/main.py --config-file config/rnet50-sgd-consine-custdataset.py --task=train --pmg
```
to the shell. 