import logging

import clip
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from PIL import Image
from torch import nn
from tqdm import tqdm
import pandas as pd
import math

accelerator = Accelerator()

from methods.unsupervised_learning_new import TextualPrompt
#new:
from utils import (compute_unlabled_logits, TrainSampler, InstanceSelector,)
import copy
from collections import defaultdict, Counter
import torch.nn.functional as F


g = torch.Generator()
g.manual_seed(0)

log = logging.getLogger(__name__)


class VisualFPL_PL(TextualPrompt):
    def __init__(
        self, 
        config,
        label_to_idx,
        data_folder,
        unlabeled_files,
        classes,
        seen_classes,
        unseen_classes,
        device,
    ):
        """This class defines few-pseudo-learning (FPL) VPT's training and evaluation.

        :param config: dictionaries of prameters in models_config/vpt_baseline_config.yml
        :param label_to_idx: dictionary (key, value):(class name, id)
        :param classes: list of class names
        :param seen_classes: list of seen classes' names
        :param unseen_classes: list of unseen classes' names
        :param device: device in use
        """
        super().__init__(
            config, label_to_idx, classes, seen_classes, unseen_classes, device
        )

        self.data_folder = data_folder

        # new added：
        self.check_unlabeled = {item: i for i, item in enumerate(unlabeled_files)}
        unseen_label_to_idx = {}
        for i in range(len(self.unseen_classes)):
            idx = self.label_to_idx[self.unseen_classes[i]]
            class_name = self.unseen_classes[i]
            unseen_label_to_idx[class_name] = idx
        self.unseen_label_to_idx = unseen_label_to_idx

        pools_idx_to_label = {i: self.unseen_classes[i] for i in range(len(self.unseen_classes))}   #{pool_idx: corresponding_label_name}
        self.Selector = InstanceSelector(label_to_idx=pools_idx_to_label, cfg=self.config, 
                                         convert_pred_method=self.convert_trzsl_pred_label)
        self.clip_model.encoder_name = self.config.VIS_ENCODER

#-----------------almost same with text_TRZSL-----------------

    def create_training_dataset(self, train_data, unlabeled_data, iter_num, epoch, true_labels=None):
        """
        Create the dataset for training including pseudolabels for unseen classes.

        Args:
            train_data (Dataset): The dataset of the training seen classes.
            unlabeled_data (Dataset): The dataset of unlabeled data for unseen classes.
            iter_num (int): The iteration number.

        Raises:
            NotImplementedError: If the learning paradigm is not 'trzsl'.

        Returns:
            Dataset, Tensor: The updated training dataset and the selected pseudolabels.
        """
        # 1. Get pseudo-labels for unlabeled data from unseen classes
        if self.config.LEARNING_PARADIGM != 'trzsl':
            raise NotImplementedError

        forward_method = self.get_clip_forward(target_class=self.unseen_classes, iter_num=iter_num, TRZSL=True)
        filepaths, probs, output_logits = compute_unlabled_logits(
            dataset=copy.deepcopy(unlabeled_data),
            transform=self.transform,
            clip_model=self.clip_model,
            forward_method=forward_method,
        )
        unlabeled_data_, PL_labels_selected = self.PPSB(
            train_data, unlabeled_data, iter_num,
            filepaths, probs
        )
        # 2. Merge unlabeled data and labeled data
        train_data_ = self._merge_unlabeled_dataset(train_data, unlabeled_data_, iter_num)
        PL_labels_selected_new = train_data_.labels.to(PL_labels_selected.device)

        return train_data_, PL_labels_selected_new


    def _merge_unlabeled_dataset(self, train_data, unlabeled_data, iter_num):
        """
        Merges unlabeled data with labeled data for TRZSL, updating the training dataset 
        with pseudo-labeled samples. 

        Args:
            train_data: The original training dataset containing labeled samples.
            unlabeled_data: The dataset containing unlabeled samples that have been
                            pseudo-labeled.
            iter_num: The current iteration number of the TRZSL process.

        Returns:
            The updated training dataset containing both originally labeled and
            pseudo-labeled samples.
        """
        # 1. get unlabeled data and labeled data for TRZSL
        all_imgs = train_data.filepaths

        train_data_ = copy.deepcopy(train_data)

        labeled_imgs = list(set(all_imgs) - set(self.check_unlabeled.keys()))
        labeled_lbs_true = self._get_gt_label(labeled_imgs, dtype=torch.long)
        labeled_lbs_true = torch.nn.functional.one_hot(labeled_lbs_true).float().cpu()

        unlabeded_imgs = unlabeled_data.filepaths
        unlabeled_pl_labels = unlabeled_data.labels

        self.balance_param = len(labeled_imgs) / len(unlabeded_imgs)
        log.info(f"UNLABELD(selected num): {len(unlabeded_imgs)} and LABELED(all num) {len(labeled_imgs)}, balance_param: {self.balance_param:.2f}")

        # 2. create a custom dataloader for labeled data:
        bs_labeled = self.balance_param * 64    #assert bs of unlabeled_data is 64:
        if bs_labeled < 8:
            bs_labeled = 8
        if bs_labeled > 256:
            bs_labeled = 256
        log.info(f"iter <{iter_num}> bs_labeled is: {bs_labeled}")
        self.train_labeled_sampler = TrainSampler(x=labeled_imgs, y=labeled_lbs_true, 
                                                bs=int(bs_labeled), 
                                                device=self.device, 
                                                transform=self.transform,
                                                mode="filepaths",
                                                img_path=labeled_imgs)
        # 3. get unlabel dataset for TRZSL:
        filepaths_new = list(unlabeded_imgs)
        labels_new = unlabeled_pl_labels
        train_data_.update_xy(labels=labels_new, filepaths=filepaths_new)

        return train_data_

    def check_partialY_acc(self, PL_labels, filepaths, target_partialR, init_partialR):
        # check the accuracy of pseudolabels
        gt_labels = self._get_gt_label(impath=filepaths, dtype=torch.long)

        # initialize a list to store the results
        results = []
        distribution = []
        # iterate over each row of PL_labels and the corresponding gt_labels
        for i in range(PL_labels.shape[0]):
            # get the indices where the values are 1.0 in the current row
            indices = torch.nonzero(PL_labels[i], as_tuple=True)

            # test if the corresponding gt_label is in these indices
            is_in = gt_labels[i] in indices[0]
            distribution.extend(indices[0].tolist())

            # append the result to the list
            results.append(is_in)
        
        results = torch.tensor(results)
        coverage_acc = results.sum() / results.shape[0]
        ct = Counter(distribution)
        ct = sorted(ct.items(), key=lambda x: x[0])
        partial_avgnum = (PL_labels > 1e-7).sum(dim=1).float()

        log.info(f"\t label_estimate_acc: {coverage_acc}")
        # log.info(f"coverage distribution: {ct}")
        partialR = partial_avgnum.mean().item()/PL_labels.shape[1]

        return {"label_estimate_acc": coverage_acc.item(), 
                "partial_ratio": partialR, 
                }


    def convert_trzsl_pred_label(self, pred_labels):
        """This function convert the predicted label of the ordinary sequence 
            to the correct TRZSL order and class names.
        """
        assert list(self.unseen_label_to_idx.keys()) == self.unseen_classes, "They should have the same order"  
        target_label_idxs = list(self.unseen_label_to_idx.values())
        if pred_labels.dim() == 2:
            converted_labels = torch.zeros((pred_labels.shape[0], len(self.classes)), 
                                                dtype=pred_labels.dtype,
                                          ).to(pred_labels.device)

            for col in range(pred_labels.shape[1]):
                target_col_idx = target_label_idxs[col]
                converted_labels[:, target_col_idx] = pred_labels[:, col]
        elif pred_labels.dim() == 1:
            converted_labels = torch.zeros(pred_labels.shape[0], 
                                           dtype=pred_labels.dtype).to(pred_labels.device) - 1
            for col in range(len(self.unseen_classes)):
                mask = (pred_labels == col)
                target_col_idx = target_label_idxs[col]
                converted_labels[mask] = target_col_idx
            
        return converted_labels


    def parser_batch(self, img, aug_1, idxs, label, img_path):
        #--------------for TRZSL----------------
        x, y = self.train_labeled_sampler.get_onebatch()
        img = torch.cat([img, x], dim=0)
        label = torch.cat([label, y], dim=0)

        return img, label
    

    def define_loss_function(self, logits, labs, idxs, paths):
        unlabeled_len = len(paths)

        loss_unseen = self.loss_func(logits[:unlabeled_len], labs[:unlabeled_len], idxs)
        loss_seen = F.cross_entropy(logits[unlabeled_len:], labs[unlabeled_len:])

        return loss_seen + self.config.LAMBDA * loss_unseen


#-----------------for visual-----------------

    def define_textual_prompts(self, target_class, only_unlabelled=None, validation=False):
        """This function returns the textual prompts. You can modify the list
        of classes of interest.

        :param only_unlabelled: boolean. It is True if the training only involves
                                pseudo-labeled unseen data
        """
        log.info(f"len of target_class for text feature is: {len(target_class)}")
        return [
            self.template.format(" ".join(i.split("_"))) for i in target_class
        ]

    def reindex_predicted_labels(self, idx_preds, only_unlabelled=False):
        """This function returns the correct index of predictions to compute
        model's accuracy.

        :param idx_pred: list of predictions ids
        :param only_unlabelled: boolean. It is True if the training only involves
                                pseudo-labeled unseen data
        """

        return [self.classes[i.item()] for i in idx_preds]

    def reindex_true_labels(self, label, only_unlabelled=False):
        """This function returns the correct index of true labels.

        :param label: list of labels from data loader
        :param only_unlabelled: boolean. It is True if the training only involves
                                pseudo-labeled unseen data
        """
        return torch.tensor([l for l in label])
    
    def PPSB(self, train_data, unlabeled_data, iter_num, filepaths, probs):
        # get image-text representation matrix W
        W = self.get_W(unlabeled_data, iter_num)

        N, M = probs.shape[0], probs.shape[1]
        labels = torch.zeros(N, M)
        labeled_st = torch.zeros(N, dtype=torch.bool)

        pred_classes = torch.argmax(probs, dim=1)
        pred_classes_np = pred_classes.cpu().numpy()

        pred_classes_unique, pred_counts = np.unique(pred_classes_np, return_counts=True)
        unseen_classes = np.arange(len(self.unseen_classes))

        pred_counts_full = np.zeros_like(unseen_classes)
        for i, cls in enumerate(unseen_classes):
            if cls in pred_classes_unique:
                pred_counts_full[i] = pred_counts[np.where(pred_classes_unique == cls)[0][0]]
        sorted_indices = np.argsort(pred_counts_full)[::-1]  
        sorted_classes = unseen_classes[sorted_indices]
        sorted_pred_counts = pred_counts_full[sorted_indices]

        head_tr = int(pred_classes.shape[0] / sorted_classes.shape[0] * self.config.tr_head)
        tail_tr = int(pred_classes.shape[0] / sorted_classes.shape[0] * self.config.tr_tail)

        head_mask = sorted_pred_counts >= head_tr
        tail_mask = sorted_pred_counts < tail_tr

        head_index = np.argmin(head_mask)
        tail_index = np.argmax(tail_mask)

        head_class = sorted_classes[:head_index]
        tail_class = sorted_classes[tail_index:]
        mid_class = sorted_classes[head_index:tail_index]
        target_class = sorted_classes[head_index:] # mid + tail

        # label the tail class
        tail_mask = np.isin(pred_classes_np, tail_class)
        tail_indices = np.where(tail_mask)[0]
        for idx in tail_indices:
            pred_cls = pred_classes_np[idx]
            labels[idx, pred_cls] = 1.0
        labeled_st[tail_indices] = True

        # label the head class
        mid_tail_total = np.sum(sorted_pred_counts[head_index:])
        Upper_bound = math.ceil(mid_tail_total / len(target_class))
        for target_head_cls in head_class:
            pred_is_head = pred_classes_np == target_head_cls
            pred_head_indices = np.where(pred_is_head)[0]
            
            pred_head_conf = probs[pred_head_indices, target_head_cls].cpu().numpy()
            
            sorted_conf_indices = np.argsort(pred_head_conf)[::-1]

            pred_head_indices_sorted = pred_head_indices[sorted_conf_indices]
            cnt = 0
            for sample_idx in pred_head_indices_sorted:
                sample_feat = W[sample_idx:sample_idx+1]  # [1, feature_dim]
                sim = F.cosine_similarity(sample_feat, W, dim=1)  # [num_samples,]
                
                sim[sample_idx] = -1.0
                top_M_sim, top_M_indices = torch.topk(sim, k=self.config.W_k, largest=True)
                top_M_indices = top_M_indices.cpu().numpy()
                
                top_M_labels = pred_classes_np[top_M_indices]
                is_consistent = (top_M_labels == target_head_cls).all()
                if cnt < Upper_bound and is_consistent:
                    labels[sample_idx][target_head_cls] = 1
                    labeled_st[sample_idx] = True
                    cnt+=1

        # label the mid class
        for target_mid_cls in mid_class:
            pred_is_mid = pred_classes_np == target_mid_cls
            pred_mid_indices = np.where(pred_is_mid)[0]

            pred_mid_conf = probs[pred_mid_indices, target_mid_cls].cpu().numpy()
            
            sorted_conf_indices = np.argsort(pred_mid_conf)[::-1]

            pred_mid_indices_sorted = pred_mid_indices[sorted_conf_indices]
            cnt = 0
            for sample_idx in pred_mid_indices_sorted:
                if cnt < Upper_bound:
                    labels[sample_idx][target_mid_cls] = 1
                    labeled_st[sample_idx] = True
                    cnt+=1

        index = np.arange(len(labeled_st))
        true_indices = [idx for idx in index if labeled_st[idx]]
        print(len(true_indices))
        filepaths_new = [filepaths[i] for i in true_indices]
        labels = labels[true_indices]
        labels = self.convert_trzsl_pred_label(labels)

        train_data_copy = copy.deepcopy(train_data)
        train_data_copy.update_xy(labels=labels.cpu(), filepaths=filepaths_new)

        return train_data_copy, labels
