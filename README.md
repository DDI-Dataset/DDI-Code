# DDI Dataset Codebase
Code for loading DDI data and the models from our paper:<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;***Disparities in Dermatology AI Performance on a Diverse, Curated Clinical Image Set***

For more information, please visit our project page [here](https://ddi-dataset.github.io/) and read our paper [here](https://www.science.org/doi/full/10.1126/sciadv.abq6147).

Our models can be downloaded [here](https://drive.google.com/drive/folders/1oQ53WH_Tp6rcLZjRp_-UBOQcMl-b1kkP) or through the provided code.


## Description 
We include code to download and load our models (`ddi_model.py`), load the DDI dataset (`ddi_dataset.py`), evaluate our models on the DDI dataset (`eval_ddi.py`) as well as evaluate our models on an arbitrary dataset  (`eval_data.py`). For `eval_ddi.py` and `eval_data.py`, we provide a command line interface with the following arguments:
- `model_dir`: File path for where to save models.
- `model`: Name of the model to load (HAM10000, DeepDerm, GroupDRO, CORAL, or CDANN).
- `no_download`: Set to disable downloading models.
- `data_dir`: Folder containing dataset to load. In `eval_ddi.py`, `data_dir` should be the root directory and contain (1) a subfolder called `images` containing all the DDI images and (2) a CSV file called `ddi_metadata.csv`. In `eval_data.py`, the structure should match the root directory in [torchvision.datasets.ImageFolder](https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.ImageFolder) with 2 classes: benign (class 0) and malignant (class 1).
- `eval_dir`: Folder to store evaluation results.
- `use_gpu`: Set to use GPU for evaluation.
- `plot`: Set to show ROC plot.


### Example usage
- Evaluate `DeepDerm` model on the DDI dataset. Data (not included in this repo) is stored in the `DDI` directory, and results will be saved in the `DDI-results` directory.
```bash
>>>python3 eval_ddi.py --model=DeepDerm --data_dir=DDI --eval_dir=DDI-results 
```
- Evaluate `DeepDerm` model on your own dataset (must be annotated as benign/malignant). Data (not included in this repo) is stored in the `MyData` directory, and results will be saved in the `DDI-results` directory.
```bash
>>>python3 eval_data.py --model=DeepDerm --data_dir=MyData --eval_dir=DDI-results 
```


## Citation
If you find this code useful or use the DDI dataset in your research, please cite:
```
@article{daneshjou2022disparities,
  title={Disparities in dermatology AI performance on a diverse, curated clinical image set},
  author={Daneshjou, Roxana and Vodrahalli, Kailas and Novoa, Roberto A and Jenkins, Melissa and Liang, Weixin and Rotemberg, Veronica and Ko, Justin and Swetter, Susan M and Bailey, Elizabeth E and Gevaert, Olivier and others},
  journal={Science advances},
  volume={8},
  number={31},
  pages={eabq6147},
  year={2022},
  publisher={American Association for the Advancement of Science}
}
```

