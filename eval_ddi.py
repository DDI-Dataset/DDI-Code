"""
Code to evaluate the models trained for our paper, 
    "Disparities in Dermatology AI Performance on a Diverse, 
     Curated Clinical Image Set",
on the DDI dataset. 

Note: assumes DDI data is organized as
    ./DDI
        /images
            /000001.png
            /000002.png
            ...
        /ddi_metadata.csv

(After downloading from the Stanford AIMI repository, this requires moving all .png files into a new subdirectory titled "images".)

------------------------------------------------------
Examples:

(1) w/command line interface
# evaluate DeepDerm on DDI and store results in `DDI-results`
>>>python3 eval_ddi.py --model=DeepDerm --data_dir=DDI --eval_dir=DDI-results 

(2) w/python functions
>>>import eval_ddi
>>>import ddi_model
>>>model = ddi_model.load_model("DeepDerm") # load DeepDerm model
>>>eval_results = eval_ddi.eval_model(model, "DDI") # evaluate images in DDI folder
"""

import argparse
from ddi_dataset import DDI_Dataset, test_transform
from ddi_model import load_model
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from sklearn.metrics import (f1_score, balanced_accuracy_score, 
    classification_report, confusion_matrix, roc_curve, auc)
import torch
import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="DDI-models", 
        help="File path for where to save models.")
    parser.add_argument('--model', type=str, default="DeepDerm", 
        help="Name of the model to load (HAM10000, DeepDerm, GroupDRO, CORAL,"\
             " or CDANN).")
    parser.add_argument('--no_download', action='store_true', default=False,
        help="Set to disable downloading models.")
    parser.add_argument('--data_dir', type=str, default="DDI", 
        help="Folder containing dataset to load. Structure should be: (1) `[data_dir]/images` contains all images; (2) `[data_dir]/ddi_metadata.csv` contains the CSV metadata for the DDI dataset")
    parser.add_argument('--eval_dir', type=str, default="DDI-results", 
        help="Folder to store evaluation results.")
    parser.add_argument('--use_gpu', action='store_true', default=False,
        help="Set to use GPU for evaluation.")
    parser.add_argument('--plot', action='store_true', default=False,
        help="Set to show ROC plot.")
    args = parser.parse_args()
    return args

def eval_model(model, dataset, use_gpu=False, show_plot=False):
    """Evaluate loaded model on provided image dataset. Assumes supplied image 
    directory corresponds to `root` input for torchvision.datasets.ImageFolder
    class. Assumes the data is split into binary/malignant labels, as this is 
    what our models are trained+evaluated on."""

    use_gpu = (use_gpu and torch.cuda.is_available())
    device = torch.device("cuda") if use_gpu else torch.device("cpu")

    # load dataset
    dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=32, shuffle=False,
                    num_workers=0, pin_memory=use_gpu)

    # prepare model for evaluation
    model.to(device).eval()

    # log output for all images in dataset
    hat, star, all_paths = [], [], []
    for batch in tqdm.tqdm(enumerate(dataloader)):
        i, (paths, images, target, skin_tone) = batch
        images = images.to(device)
        target = target.to(device)

        with torch.no_grad():
            output = model(images)

        hat.append(output[:,1].detach().cpu().numpy())
        star.append(target.cpu().numpy())
        all_paths.append(paths)

    hat = np.concatenate(hat)
    star = np.concatenate(star)
    all_paths = np.concatenate(all_paths)
    threshold = model._ddi_threshold
    m_name = model._ddi_name
    m_web_path = model._ddi_web_path

    report = classification_report(star, (hat>threshold).astype(int), 
        target_names=["benign","malignant"])
    fpr, tpr, _ = roc_curve(star, hat, pos_label=1,
                                sample_weight=None,
                                drop_intermediate=True)
    auc_est = auc(fpr, tpr)

    if show_plot:
        _=plt.plot(fpr, tpr, 
            color="blue", linestyle="-", linewidth=2, 
            marker="o", markersize=2, 
            label=f"AUC={auc_est:.3f}")[0]
        plt.show()
        plt.close()

    eval_results = {'predicted_labels':hat, # predicted labels by model
                    'true_labels':star,     # true labels
                    'images':all_paths,     # image paths
                    'report':report,        # sklearn classification report
                    'ROC_AUC':auc_est,      # ROC-AUC
                    'threshold':threshold,  # >= threshold ==> malignant
                    'model':m_name,         # model name
                    'web_path':m_web_path,  # web link to download model
                    }

    return eval_results




if __name__ == '__main__':
    # get arguments from command line
    args = get_args()
    # load model and download if necessary
    model = load_model(args.model, 
        save_dir=args.model_dir, download=not args.no_download)
    # load DDI dataset
    dataset = DDI_Dataset("DDI", transform=test_transform)
    # evaluate results on data
    eval_results = eval_model(model, dataset, 
        use_gpu=args.use_gpu, show_plot=args.plot)

    # save evaluation results in a .pkl file 
    if args.eval_dir:
        os.makedirs(args.eval_dir, exist_ok=True)
        eval_save_path = os.path.join(args.eval_dir, 
                                      f"{args.model}-evaluation.pkl")
        with open(eval_save_path, 'wb') as f:
            pickle.dump(eval_results, f)

        # load results with:
        #with open(eval_save_path, 'rb') as f:
        #    results = pickle.load(f)
