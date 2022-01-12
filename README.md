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
source .bashrc/.zshrc
```
respectively. You are good to go. 

# Getting Started
ThuDogs runs all of its training and testing through ```config-file```. Please refer to [config.md](docs/config.md) for details. 