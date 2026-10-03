import copy
from functools import reduce
import logging
import math
import nltk
from nltk.corpus import wordnet as wn
from operator import mul

import clip
import numpy as np
import pandas as pd
import scipy.stats as st
import torch
from accelerate import Accelerator
from PIL import Image
from torch import nn
from torch.nn.modules.utils import _pair
from torchvision import transforms
import re
import copy
from utils.loss import PLL_loss
from tqdm import tqdm
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_kernels
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter
from torch.nn.functional import cosine_similarity
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import sys
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

from dassl.metrics import compute_accuracy
from dassl.utils import (
    AverageMeter, 
)

accelerator = Accelerator()

from models import (
    CustomImageEncoder, 
    CustomTextEncoder, 
    ImagePrefixModel,
    TextPrefixModel,
    UPTModel,
)
from utils import (
    make_scheduler, 
    seed_worker, 
    save_parameters,
    save_pseudo_labels,
    save_pseudo_labels_torch,
)


g = torch.Generator()
g.manual_seed(0)

log = logging.getLogger(__name__)

class TrainingStrategy(object):
    def __init__(
        self, 
        config, 
        label_to_idx, 
        classes, 
        seen_classes, 
        unseen_classes, 
        device
    ):
        """ This class defines functions for the training strategies.

        :param config: dictionaries of prameters in models_config/vpt_baseline_config.yml
        :param label_to_idx: dictionary (key, value):(class name, id)
        :param classes: list of class names
        :param seen_classes: list of seen classes' names
        :param unseen_classes: list of unseen classes' names
        :param device: device in use
        """

        self.config = config
        self.classes = classes
        self.seen_classes = seen_classes
        self.unseen_classes = unseen_classes
        self.label_to_idx = label_to_idx

        self.device = device
        # OpenAI CLIP
        self.clip_model, self.transform = clip.load(
            self.config.VIS_ENCODER, device=self.device
        )

        # OpenCLIP
        # self.clip_model, self.transform = clip.load(
        #     self.config.VIS_ENCODER, 
        #     device=self.device,
        #     jit=False,
        #     download_root=None
        # )
        # checkpoint = torch.load("./OpenCLIP_models/vit_b_32-laion2b.pth", map_location=self.device)
        # self.clip_model.load_state_dict(checkpoint)

        self.transform_train = self.modify_transform(self.transform)
        self.template = self.config.PROMPT_TEMPLATE
        self.descriptions = self.config.DATASET_DESCRIPTIONS
        self.get_W_matrix = None

    def modify_transform(self, transform):
        """
        Modify an existing transform.
        
        Parameters:
        transform (torchvision.transforms.Compose): The existing transform
    
        Returns:
        torchvision.transforms.Compose: The modified transform
        """
        # Get the normalization transform from the existing transform
        normalize = [t for t in transform.transforms if isinstance(t, transforms.Normalize)][0]
        # Get the Resize transform from the existing transform
        resize_transform = [t for t in transform.transforms if isinstance(t, transforms.CenterCrop)][0]
        # Parse the size from the Resize transform's print information
        size_info = re.search(r'size=\((\d+), (\d+)\)', str(resize_transform))
        H, W = map(int, size_info.groups())

        # Build the new transform
        transform_new = transforms.Compose([
            transforms.RandomResizedCrop(size=(H, W), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize  # Use the same normalization as the existing transform
        ])
        
        return transform_new


    def declare_custom_encoder(self):
        """ This function declares the custom encoder
        needed depending on the prompt modality.

        :param modality: either text or image
        """

        if self.config.MODALITY == 'image':
            self.visual_transformer = self.clip_model.visual
            self.image_encoder = CustomImageEncoder(self.visual_transformer
            ).to(self.device)
            log.info(f"Freeze visual encoder.")
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        elif self.config.MODALITY == 'text':
            if torch.cuda.is_available():
                self.text_encoder = CustomTextEncoder(
                    self.clip_model, self.device, torch.float16
                ).to(self.device)
            else:
                self.text_encoder = CustomTextEncoder(
                    self.clip_model, self.device, torch.half
                ).to(self.device)

            log.info(f"Freeze text encoder.")
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        elif self.config.MODALITY == 'multi':
            self.visual_transformer = self.clip_model.visual
            self.image_encoder = CustomImageEncoder(self.visual_transformer).to(self.device)
            if torch.cuda.is_available():
                self.text_encoder = CustomTextEncoder(
                    self.clip_model, self.device, torch.float16
                ).to(self.device)
            else:
                self.text_encoder = CustomTextEncoder(
                    self.clip_model, self.device, torch.half
                ).to(self.device)

            log.info(f"Freeze visual encoder.")
            for param in self.image_encoder.parameters():
                param.requires_grad = False

            log.info(f"Freeze text encoder.")
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def initialize_prompts_parameters(self):
        """ This function initialized the prompt parameters
        depending on the prompt modality.

        :param modality: either text or image
        """

        if self.config.MODALITY == 'image':
            width = self.visual_transformer.class_embedding.size()[0]
            scale = width**-0.5
            if self.config.VIS_PREFIX_INIT == "normal":
                vis_initial_prefix = scale * torch.randn(self.config.PREFIX_SIZE, width)

            elif self.config.VIS_PREFIX_INIT == "uniform":
                val = math.sqrt(6.0 / float(3 * reduce(mul, (16, 16), 1) + width))  # noqa
                vis_initial_prefix = torch.zeros(self.config.PREFIX_SIZE, width)
                vis_initial_prefix = scale * nn.init.uniform_(vis_initial_prefix, -val, val)

            self.vis_initial_prefix = vis_initial_prefix

        elif self.config.MODALITY == 'text':
            # Prefix initialization
            prefix_dim = (
                1,
                self.config.PREFIX_SIZE,
                self.clip_model.token_embedding.embedding_dim,
            )
            self.initial_prefix = torch.normal(         #TODO set prompt init
                self.config.MEAN_INIT, self.config.VAR_INIT, size=prefix_dim
            ).to(self.device)

        elif self.config.MODALITY == 'multi':
            # Get relevant dimensions
            vpt_dim = self.clip_model.visual.conv1.weight.shape[0]
            coop_dim = self.clip_model.ln_final.weight.shape[0]

            # Initialize the coop prompt
            self.coop_embeddings = torch.empty(
                1, 
                self.config.TEXT_PREFIX_SIZE, 
                coop_dim,
                dtype=self.dtype).to(self.device)
            nn.init.normal_(self.coop_embeddings, std=0.02)

            # Initialize the vpt prompt
            clip_patchsize = self.clip_model.visual.conv1.weight.shape[-1]
            clip_patchsize = _pair(clip_patchsize)
            val = math.sqrt(6. / float(3 * reduce(mul, clip_patchsize, 1) + vpt_dim))  # noqa

            self.vpt_embeddings = torch.zeros(
                1, 
                self.config.VISION_PREFIX_SIZE, 
                vpt_dim, 
                dtype=self.dtype).to(self.device)
            # xavier_uniform initialization
            nn.init.uniform_(self.vpt_embeddings.data, -val, val)

            if self.config.VPT_DEEP:
                self.vision_layers = len([k for k in self.clip_model.state_dict().keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])

                self.vpt_embeddings_deep = torch.zeros(
                        self.vision_layers-1, 
                        self.config.VISION_PREFIX_SIZE, 
                        vpt_dim, 
                        dtype=self.dtype).to(self.device)
                # xavier_uniform initialization
                nn.init.uniform_(self.vpt_embeddings_deep.data, -val, val)
            else:
                self.vpt_embeddings_deep = None


    def define_model(self, classes=None):
        """ This function initialized the model
        depending on the prompt modality.

        :param modality: either text or image
        :param classes: the list of classes for textual model
        """

        if self.config.MODALITY == 'image':
            # Define model
            self.model = ImagePrefixModel(
                copy.deepcopy(self.vis_initial_prefix),
                self.image_encoder,
                device=self.device,
            ).to(self.device)

        elif self.config.MODALITY == 'text':
            # Define model
            self.model = TextPrefixModel(
                copy.deepcopy(self.initial_prefix),    #torch.Size([1, 16, 512])
                self.text_encoder,
                [" ".join(c.split("_")) for c in classes],
                device=self.device, 
                descriptions=self.descriptions,
            ).to(self.device)

        elif self.config.MODALITY == 'multi':

            # Define model
            self.model = UPTModel(
                self.coop_embeddings,
                self.vpt_embeddings,
                self.vpt_embeddings_deep,
                self.image_encoder,
                self.text_encoder,
                self.classes,
                self.config.TRANSFORMER_DIM, 
                device=self.device,
                dtype=self.dtype
            ).to(self.device)

        for i, parameter in enumerate(self.model.parameters()):
            if parameter.requires_grad:
                log.info(f"Shape of parameters {i}: {parameter.shape}")

        if self.config.OPTIM == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.LR,
                weight_decay=self.config.DECAY,
                momentum=0.9,
            )

        self.scheduler = make_scheduler(self.optimizer, self.config)
    
    def build_loss(self, cfg, partialY):
        if cfg.LOSS_TYPE == 'CE':
            criterion = torch.nn.CrossEntropyLoss()     
            criterion.cfg = cfg
        else:
            if cfg.HAS_CONF:
                criterion = PLL_loss(type=cfg.LOSS_TYPE, cfg=cfg,
                                     PartialY=copy.deepcopy(partialY))
            else:
                criterion = PLL_loss(type=cfg.LOSS_TYPE, cfg=cfg,
                                     PartialY=copy.deepcopy(partialY))
        self.loss_func = criterion
        return self.loss_func


    def create_training_dataset(self, train_data, unlabeled_data=None):
        """This function create the dataset for training. Specifically, it
        merges pseudo-labels for unseen data and labeled data for seen classes.

        :param train_data: Dataset object - training seen classes (defined in zsl_jpl line 323)
        :param unlabeled_data: Dataset object - dataset of unlabeled data for
                               unseen classes (defined in zsl_jpl line 328)
        """
        self.val_unseen_files = None
        return train_data


    def _before_train(self, train_data, val_data=None, train_transform=None, val_transform=None):
        # Declare the data pre processing for train and validation data
        train_data.transform = train_transform
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
            num_workers=8,
            drop_last=True,
            pin_memory=(torch.cuda.is_available()),
        )
        if val_data is not None:
            val_data.transform = val_transform
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.config.BATCH_SIZE,
                shuffle=False,
                num_workers=8,
                drop_last=False,
            )
        else:
            val_loader = None
        
        accelerator.wait_for_everyone()

        self.model, self.optimizer, train_loader, val_loader = accelerator.prepare(
            self.model, self.optimizer, train_loader, val_loader)
        if val_loader is not None:
            log.info(f"Size of validation dataset: {len(val_data.filepaths)}")
        
        return train_loader, val_loader

    def train(
        self,
        train_data,
        unlabeled_data,
        val_data=None,
        only_unlabelled=False,
        only_seen=False,
        iter_num=None,
        test_dataset=None
    ):
        """This function defines the current training iteration of self.model.

        Args:
            train_data (CustomDataset): The labeled training dataset.
            unlabeled_data (CustomDataset): The unlabeled dataset.
            val_data (CustomDataset, optional): The validation dataset. Default is None.
            only_unlabelled (bool, optional): If True, train only with unlabeled data. Default is False.
            only_seen (bool, optional): If True, train only with seen classes. Default is False.
            iter_num (int, optional): The current iteration number. Default is None.
        """
        # 1. Define training dataset, model and loss
        log.info(f"Current train data (all num can be used) is: {len(train_data.filepaths)}.")
        log.info(f"And only with unlabeled is: {len(unlabeled_data.filepaths)}.")
        log.info(f"We select max {self.config.N_PSEUDOSHOTS} pseudolabel per each unseen classes.")

        if self.config.LEARNING_PARADIGM == 'trzsl':
            current_pl_classes = self.unseen_classes
            log.info(f"[TRZSL] The number of unseen classes/all classes is: "
                     f"{len(self.unseen_classes)}/{len(self.classes)}.")
        else:
            current_pl_classes = self.classes
        log.info(f"Thus we expect an max number of pseudolabeles equal to "
                 f"{len(current_pl_classes) * self.config.N_PSEUDOSHOTS}.")

        log.info(f"Start generating train_dataset..")
        
        self.image_prototypes = None

        # Create the current training dataset from unlabeled_data
        train_data_, plabels = self.create_training_dataset(
            train_data, unlabeled_data, 
            iter_num, 0
        )

        # Initialize the model
        log.info(f"Model Initialization..")
        if self.config.MODALITY == 'text':
            self.define_model(self.classes)
        else:
            self.define_model()

        #2. prepare train loader
        train_loader, val_loader = self._before_train(
            train_data_, val_data, 
            train_transform=self.transform, 
            val_transform=self.transform
        )

        self.image_prototypes = self.get_image_prototypes(train_loader, iter_num=1)

        # regenerate pseudo-labels
        train_data_, plabels = self.create_training_dataset(
            train_data, unlabeled_data, 
            iter_num, 0
        )
        train_loader, val_loader = self._before_train(
            train_data_, val_data, 
            train_transform=self.transform, 
            val_transform=self.transform
        )

        # Initialize the loss function
        log.info(f"Building loss function..")
        self.loss_func = self.build_loss(self.config.LOSS_CFG, plabels)

        best_val_accuracy = 0; loss = None

        # 3. start training:
        for epoch in range(self.config.EPOCHS):
            log.info(f"Run Epoch {epoch}")
            if epoch !=0 and epoch % 5 == 0:
                with torch.no_grad():
                    std_predictions = self.test_predictions(test_dataset, standard_zsl=True)
                    log.info(f"Testset accuracy: {std_predictions}")

                    train_data_, plabels = self.create_training_dataset(train_data, unlabeled_data, 2, epoch)
                    train_loader, val_loader = self._before_train(
                        train_data_, val_data, 
                        train_transform=self.transform, 
                        val_transform=self.transform
                    )
                    self.image_prototypes = self.get_image_prototypes(train_loader, iter_num=2)
                    train_data_, plabels = self.create_training_dataset(train_data, unlabeled_data, 2, epoch)
                    train_loader, val_loader = self._before_train(
                        train_data_, val_data, 
                        train_transform=self.transform, 
                        val_transform=self.transform
                    )

            total_loss = 0
            accum_iter = self.config.ACCUMULATION_ITER

            loss, total_loss, epoch_parameters = self._train_epoch(
                loss,
                total_loss,
                train_loader,
                accum_iter,
                epoch,
                only_unlabelled=only_unlabelled,
                only_seen=only_seen,
            )
            accelerator.wait_for_everyone()
            self._after_epoch(                
                train_data,
                epoch,)

            if accelerator.is_local_main_process:
                log.info(f"Loss Epoch {epoch}: {total_loss/(len(train_loader))}")

            if val_loader is not None:
                val_accuracy = self._run_validation(val_loader, only_unlabelled)
                log.info(f"Validation accuracy after Epoch {epoch}: {val_accuracy}")
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_prompt = epoch_parameters
            else:
                best_val_accuracy = None
                best_prompt = epoch_parameters

            if self.config.MODALITY == 'text':
                # After validation on seen classes redefine the set of training classes
                self.model.classes = self.classes
        
        

        return best_val_accuracy, epoch_parameters


    @torch.no_grad()
    def _after_epoch(self, train_data, epoch):
        if not hasattr(self.loss_func, 'losstype') or '_' not in self.loss_func.losstype:
            """the loss_func do not need post-epoch processing (update conf)"""
            return

        elif epoch >= 0:
            train_loader, val_loader = self._before_train(train_data, val_data=None, 
                                                          train_transform=self.transform)

            acc_cum = AverageMeter()
            forward_method = self.get_clip_forward(target_class=self.classes)
            for i, (img, aug_1, idxs, label, img_path) in enumerate(train_loader):
                gt_label = self._get_gt_label(img_path, dtype=label.dtype)

                logits = forward_method(img)
                self.loss_func.check_conf_update(img, label, idxs, output=logits)   

                acc_cum.update(compute_accuracy(logits, gt_label)[0].item())
                if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
                    log.info(
                        f"EVAL on epoch [{epoch}/{self.config.EPOCHS}] [{(i + 1)}/{len(train_loader)}]\t" 
                        f"acc {acc_cum.val:.3f} ({acc_cum.avg:.3f})\t"
                    )

            self.loss_func.clean_conf()


    def define_loss_function(self, logits, labs):
        """return the loss function value."""
        return self.loss_func(logits, labs)

    def detect_anomaly(self, loss):
        if not torch.isfinite(loss).all():
            log.info(f"Loss is infinite or NaN!")
            raise FloatingPointError("Loss is infinite or NaN!")

    def backpropagate(self):
        self.optimizer.step()
        self.model.zero_grad()

    def update_scheduler(self):
        current_lr = self.scheduler.get_last_lr()
        self.scheduler.step()

    def unwrap_model(self):
        return accelerator.unwrap_model(self.model)
    
    def get_image_prototypes(self, train_loader, dtype=torch.float32, iter_num=1):
        """Get image prototypes for all classes (seen and unseen) based on CLIP features.

        Args:
            unlabeled_data (CustomDataset): The unlabeled dataset containing images from both seen and unseen classes.
        """
        image_features = []
        logits_list = []
        forward_method = self.get_clip_forward(target_class=self.classes, iter_num=iter_num) 

        for img, aug_1, idxs, label, img_path in tqdm(train_loader):
            with torch.no_grad():
                img = img.to(self.device)
                if self.config.MODALITY == 'text':
                    features = img.type(dtype)
                elif self.config.MODALITY == 'image':
                    features = self.model(img.to(self.device)).type(dtype)
                image_features.append(features)
                logits = forward_method(img)
                logits_list.append(logits)

        logits_all = torch.cat(logits_list, dim=0)
        probs = torch.softmax(logits_all, dim=1)
        pred_classes = torch.argmax(probs, dim=1)

        image_features = torch.cat(image_features, dim=0)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        unique_labels = torch.sort(torch.unique(pred_classes))[0]
        prototypes = []

        for label in torch.arange(len(self.classes)):
            if label in unique_labels:
                mask = (pred_classes == label)
                class_features = image_features[mask]
                prototype = torch.mean(class_features, dim=0)
                prototypes.append(prototype)
            else:
                class_probs = probs[:, label]
                topk_indices = torch.topk(class_probs, k=min(8, len(class_probs)), dim=0).indices
                topk_features = image_features[topk_indices]
                prototype = torch.mean(topk_features, dim=0)
                prototypes.append(prototype)

        prototypes = torch.stack(prototypes, dim=0)
        prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)
        
        return prototypes

    def get_W(self, unlabeled_data, prototypes=None, dtype=torch.float32, iter_num=1):
        print("start solve W")
        gamma = self.config.W_gamma

        if self.config.MODALITY == 'text':
            if self.get_W_matrix == None:
                dataset = copy.deepcopy(unlabeled_data)
                dataset.transform = self.transform
                train_loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=self.config.BATCH_SIZE,
                    shuffle=False,
                    num_workers=8,
                    drop_last=False,
                    pin_memory=True if torch.cuda.is_available() else False,
                )
                X = []
                for img, aug_1, idxs, label, img_path in tqdm(train_loader):
                    with torch.no_grad():
                        if img.dim() == 4:
                            image_features = self.clip_model.encode_image(img.to(self.device))
                            image_features = image_features / image_features.norm(
                                    dim=-1, keepdim=True).type(dtype)
                        elif img.dim() == 2:
                            image_features = img.to(self.device).type(dtype)
                        X.append(image_features)
                X = torch.cat(X, dim=0)
                X = X / X.norm(dim=-1, keepdim=True)
                self.get_W_matrix = X
            else:
                X = self.get_W_matrix
        
        elif self.config.MODALITY == 'image':
            dataset = copy.deepcopy(unlabeled_data)
            dataset.transform = self.transform
            train_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=False,
                num_workers=8,
                drop_last=False,
                pin_memory=True if torch.cuda.is_available() else False,
            )
            X = []
            for img, aug_1, idxs, label, img_path in tqdm(train_loader):
                with torch.no_grad():
                    if img.dim() == 4:
                        if iter_num == 1:
                            image_features = self.clip_model.encode_image(img.to(self.device)).type(dtype)
                        else:
                            image_features = self.model(img.to(self.device)).type(dtype)
                    X.append(image_features)
            X = torch.cat(X, dim=0)
            X = X / X.norm(dim=-1, keepdim=True)
        
        if self.config.MODALITY == 'text':
            with torch.no_grad():
                if iter_num == 1:
                    if self.descriptions is None:
                        prompts = [self.config.PROMPT_TEMPLATE.format(c.replace("_", " ")) for c in self.classes]
                    else:
                        prompts = [self.descriptions[c.replace("_", " ")][0] for c in self.classes]
                    class_embeddings = clip.tokenize(prompts).to(self.device)
                    C = self.clip_model.encode_text(class_embeddings).type(dtype)
                else:
                    prompts = [c.replace("_", " ") for c in self.classes]
                    C = self.model(prompts).type(dtype) 
        elif self.config.MODALITY == 'image':
            with torch.no_grad():
                if self.descriptions is None:
                    prompts = [self.config.PROMPT_TEMPLATE.format(c.replace("_", " ")) for c in self.classes]
                else:
                    prompts = [self.descriptions[c.replace("_", " ")][0] for c in self.classes]
                class_embeddings = clip.tokenize(prompts).to(self.device)
                C = self.clip_model.encode_text(class_embeddings).type(dtype)
        
        
        C = C / C.norm(dim=-1, keepdim=True)
        
        X = X.to(self.device)
        # C = X @ C.T @ (C @ C.T + gamma * I_M)^{-1}
        # C -> shape[sample_num, word_num]  
        # X -> shape[sample_num, dim]  
        # C -> shape[word_num, dim]
        M = C.size(0)
        CCt = C @ C.T
        reg_term = gamma * torch.eye(M, device=self.device, dtype=C.dtype)
        A = CCt + reg_term
        XCt = X @ C.T
        C = XCt @ torch.linalg.inv(A)

        return C