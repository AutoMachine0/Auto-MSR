# AutoMSR

- AutoMSR is an molecular structure automatic representation learning framework for multi-label metabolic pathway prediction.

- The framework of AutoMSR is as follows:

<br>
<div align=left> <img src="pic/AutoMSR.svg" height="100%" width="100%"/> </div>


## News

- 2021.10.24 Our work [*AutoMSR: Auto Molecular Structure Representation Learning for Multi-label Metabolic Pathway Prediction*](https://ieeexplore.ieee.org/document/9864145) is accepted by **TCBB 2022**.

  
## Installing For Ubuntu 16.04

- **Ensure you have installed CUDA 10.2 before installing other packages**

**1. Nvidia and CUDA 10.2:**

```python
[Nvidia Driver] 
https://www.nvidia.cn/Download/index.aspx?lang=cn

[CUDA 10.2 Download and install command] 
#download:
wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
#install:
sudo sh cuda_10.2.89_440.33.01_linux.run

```

**2. Python environment:** recommending using Conda package manager to install

```python
conda create -n automsr python=3.7
source activate automsr
```

**3. Pytorch 1.8.1:** execute the following command in your conda env automsr

```python
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

**4. Pytorch Geometric 2.0.2:** execute the following command in your conda env automsr
```python
pip install torch-scatter==2.0.9 torch-sparse==0.6.12 torch-cluster==1.5.9 torch-spline-conv==1.2.1 torch-geometric==2.0.2 -f https://data.pyg.org/whl/torch-1.8.0+cu102.html
```
**5. Ray 1.7.0:** execute the following command in your conda env automsr
```python
pip install ray==1.7.0
```

## Running the Experiment
**1.Searching the GNN Architecture and Hyper Parameter**
```python
python search_main.py
```

**2.Testing the Optimal Model Designed by AutoMSR**
```python
python test_main.py
```

## Citing

If you think AutoMSR is useful tool for you, please cite our paper, thank you for your support:
```python
@article{chen2022automsr,
  title={AutoMSR: Auto Molecular Structure Representation Learning for Multi-label Metabolic Pathway Prediction},
  author={Chen, Jiamin and Gao, Jianliang and Lyu, Tengfei and Oloulade, Babatounde Moctard and Hu, Xiaohua},
  journal={IEEE/ACM Transactions on Computational Biology and Bioinformatics},
  number={01},
  pages={1--11},
  year={2022},
  publisher={IEEE Computer Society}
}
```
