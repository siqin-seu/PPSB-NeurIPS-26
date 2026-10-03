import json
import logging
import os
import random

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def get_class_names(dataset, dataset_dir, seed=500):
    """Returns the lists of the names of all classes, seen classes,
    and unseen classes.

    :param dataset: name of the dataset in use
    :param dataset_dir: path to get dataset dir (on CCV)
    """
    if dataset == "DTD":
        path = f"{dataset_dir}/{dataset}"

        classes = []
        with open(f"{path}/class_names.txt", "r") as f:
            for l in f:
                classes.append(l.strip())

        np.random.seed(seed)
        seen_indices = np.random.choice(
            range(len(classes)), size=int(len(classes) * 0.62), replace=False
        )
        unseen_indices = list(set(range(len(classes))).difference(set(seen_indices)))

        seen_classes = list(np.array(classes)[seen_indices])
        unseen_classes = list(np.array(classes)[unseen_indices])

    elif dataset == "FGVCAircraft":
        path = f"{dataset_dir}/{dataset}"

        classes = []
        with open(f"{path}/labels.txt", "r") as f:
            for l in f:
                classes.append(l.strip())

        np.random.seed(seed)
        seen_indices = np.random.choice(
            range(len(classes)), size=int(len(classes) * 0.62), replace=False
        )
        unseen_indices = list(set(range(len(classes))).difference(set(seen_indices)))

        seen_classes = list(np.array(classes)[seen_indices])
        unseen_classes = list(np.array(classes)[unseen_indices])

    elif dataset == "Flowers102":
        path = f"{dataset_dir}/{dataset}"

        classes = []
        with open(f"{path}/class_names.txt", "r") as f:
            for l in f:
                classes.append(l.strip())

        np.random.seed(seed)
        seen_indices = np.random.choice(
            range(len(classes)), size=int(len(classes) * 0.62), replace=False
        )
        unseen_indices = list(set(range(len(classes))).difference(set(seen_indices)))

        seen_classes = list(np.array(classes)[seen_indices])
        unseen_classes = list(np.array(classes)[unseen_indices])

    elif dataset == "CUB":
        path = f"{dataset_dir}/{dataset}"

        classes = []
        with open(f"{path}/CUB_200_2011/classes.txt", "r") as f:
            for l in f:
                classes.append(l.strip().split(".")[-1].replace("_", " ").lower())

        np.random.seed(seed)
        seen_indices = np.random.choice(
            range(len(classes)), size=int(len(classes) * 0.62), replace=False
        )
        unseen_indices = list(set(range(len(classes))).difference(set(seen_indices)))

        seen_classes = list(np.array(classes)[seen_indices])
        unseen_classes = list(np.array(classes)[unseen_indices])
    
    elif dataset == "OxfordPets":
        path = f"{dataset_dir}/{dataset}"

        classes = []
        with open(f"{path}/class_names.txt", "r") as f:
            for l in f:
                classes.append(l.strip())

        np.random.seed(seed)
        seen_indices = np.random.choice(
            range(len(classes)), size=int(len(classes) * 0.62), replace=False
        )
        unseen_indices = list(set(range(len(classes))).difference(set(seen_indices)))

        seen_classes = list(np.array(classes)[seen_indices])
        unseen_classes = list(np.array(classes)[unseen_indices])
    
    elif dataset == "StanfordCars":
        path = f"{dataset_dir}/{dataset}"

        classes = []
        with open(f"{path}/class_names.txt", "r") as f:
            for l in f:
                classes.append(l.strip())

        np.random.seed(seed)
        seen_indices = np.random.choice(
            range(len(classes)), size=int(len(classes) * 0.62), replace=False
        )
        unseen_indices = list(set(range(len(classes))).difference(set(seen_indices)))

        seen_classes = list(np.array(classes)[seen_indices])
        unseen_classes = list(np.array(classes)[unseen_indices])

    return classes, seen_classes, unseen_classes


def get_labeled_and_unlabeled_data(
    dataset, data_folder, seen_classes, unseen_classes, classes=None
):
    """This function returns the list of
    - labeled_data: each item is (image name, class name)
    - unlabeled_data: each item is (image name, class name)
    - test_data: each item is (image name, class name)

    :param dataset: dataset name
    :param data_folder: path to folder of images
    :param seen_classes: list of seen classes' names
    :param unseen_classes: list of unseen classes' names
    """
    if dataset == "DTD":
        labeled_files = []
        labels_files = []

        unlabeled_lab_files = []
        unlabeled_labs = []

        for split in ["train", "val"]:
            with open(f"{data_folder}/{split}.txt", "r") as f:
                for l in f:
                    line = l.split(" ")
                    cl = classes[int(line[1].strip())]  #cl is label form val set(true label)
                    if cl in seen_classes:
                        labeled_files.append(
                            f"{split}/{line[0].strip().split('@')[-1]}"
                        )
                        labels_files.append(cl)
                    elif cl in unseen_classes:
                        unlabeled_lab_files.append(
                            f"{split}/{line[0].strip().split('@')[-1]}"
                        )
                        unlabeled_labs.append(cl)
                    else:
                        raise Exception(
                            f"The extracted class is not among the seen or unseen classes."
                        )

        labeled_data = list(zip(labeled_files, labels_files))
        unlabeled_data = list(zip(unlabeled_lab_files, unlabeled_labs))

        test_files = []
        test_labs = []

        with open(f"{data_folder}/test.txt", "r") as f:
            for l in f:
                line = l.split(" ")
                test_files.append(f"test/{line[0].strip().split('@')[-1]}")
                test_labs.append(classes[int(line[1].strip())])
        test_data = list(zip(test_files, test_labs))

        return labeled_data, unlabeled_data, test_data

    elif dataset == "FGVCAircraft":
        labeled_files = []
        labels_files = []

        unlabeled_lab_files = []
        unlabeled_labs = []

        for split in ["train", "val"]:
            with open(f"{data_folder}/{split}.txt", "r") as f:
                for l in f:
                    img = " ".join(l.split(" ")[:-1]).split("@")[-1].strip()
                    cl = img.split("/")[0].strip()

                    if cl in seen_classes:
                        labeled_files.append(f"{split}/{img}")
                        labels_files.append(cl)
                    elif cl in unseen_classes:
                        unlabeled_lab_files.append(f"{split}/{img}")
                        unlabeled_labs.append(cl)
                    else:
                        raise Exception(
                            f"The extracted class {cl} is not among the seen or unseen classes."
                        )

        labeled_data = list(zip(labeled_files, labels_files))
        unlabeled_data = list(zip(unlabeled_lab_files, unlabeled_labs))

        test_files = []
        test_labs = []

        with open(f"{data_folder}/test.txt", "r") as f:
            for l in f:
                img = " ".join(l.split(" ")[:-1]).split("@")[-1].strip()
                cl = img.split("/")[0].strip()

                test_files.append(f"test/{img}")
                test_labs.append(cl)

        test_data = list(zip(test_files, test_labs))

        return labeled_data, unlabeled_data, test_data

    elif dataset == "Flowers102":
        labeled_files = []
        labels_files = []

        unlabeled_lab_files = []
        unlabeled_labs = []

        for split in ["train", "val"]:
            with open(f"{data_folder}/{split}.txt", "r") as f:
                for l in f:
                    line = l.split(" ")
                    img = line[0].split("@")[-1].strip()
                    cl = classes[int(line[1].strip())]

                    if cl in seen_classes:
                        labeled_files.append(f"{split}/{img}")
                        labels_files.append(cl)
                    elif cl in unseen_classes:
                        unlabeled_lab_files.append(f"{split}/{img}")
                        unlabeled_labs.append(cl)
                    else:
                        raise Exception(
                            f"The extracted class {cl} is not among the seen or unseen classes."
                        )

        labeled_data = list(zip(labeled_files, labels_files))
        unlabeled_data = list(zip(unlabeled_lab_files, unlabeled_labs))

        test_files = []
        test_labs = []

        with open(f"{data_folder}/test.txt", "r") as f:
            for l in f:
                line = l.split(" ")
                img = line[0].split("@")[-1].strip()
                cl = classes[int(line[1].strip())]

                test_files.append(f"test/{img}")
                test_labs.append(cl)

        test_data = list(zip(test_files, test_labs))

        return labeled_data, unlabeled_data, test_data

    elif dataset == "CUB":
        labeled_files = []
        labels_files = []

        unlabeled_lab_files = []
        unlabeled_labs = []

        img_name_list=[]
        img_txt_file = open(os.path.join(data_folder, 'CUB_200_2011', 'images.txt'))
        train_val_file = open(os.path.join(data_folder, 'CUB_200_2011', 'train_test_split.txt'))

        for line in img_txt_file:
            # The last character is a line break
            img_name_list.append(line[:-1].split(' ')[-1])

        # Set up training and testing sets
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))

        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]


        for l in train_file_list:
            line = l.strip()
            cl = line.split("/")[0].split(".")[-1].strip().replace("_", " ").lower()
            if cl in seen_classes:
                labeled_files.append(f"CUB_200_2011/images/{line}")
                labels_files.append(cl)
            elif cl in unseen_classes:
                unlabeled_lab_files.append(f"CUB_200_2011/images/{line}")
                unlabeled_labs.append(cl)
            else:
                raise Exception(
                    f"The extracted class is not among the seen or unseen classes."
                )

        labeled_data = list(zip(labeled_files, labels_files))
        unlabeled_data = list(zip(unlabeled_lab_files, unlabeled_labs))

        test_files = []
        test_labs = []

        for l in test_file_list:
            line = l.strip()
            test_files.append(f"CUB_200_2011/images/{line}")
            test_labs.append(
                line.split("/")[0].split(".")[-1].strip().replace("_", " ").lower()
            )
        test_data = list(zip(test_files, test_labs))

        return labeled_data, unlabeled_data, test_data
    
    elif dataset == "OxfordPets":
        labeled_files = []
        labels_files = []

        unlabeled_lab_files = []
        unlabeled_labs = []

        with open(f"{data_folder}/train.txt", "r") as f:
            for l in f:
                line = l.rsplit(" ",1)
                img = line[0].strip()
                cl = classes[int(line[-1].strip())]

                if cl in seen_classes:
                    labeled_files.append(f"{img}")
                    labels_files.append(cl)
                elif cl in unseen_classes:
                    unlabeled_lab_files.append(f"{img}")
                    unlabeled_labs.append(cl)
                else:
                    raise Exception(
                        f"The extracted class {cl} is not among the seen or unseen classes."
                    )
                
        labeled_data = list(zip(labeled_files, labels_files))
        unlabeled_data = list(zip(unlabeled_lab_files, unlabeled_labs))

        test_files = []
        test_labs = []

        with open(f"{data_folder}/test.txt", "r") as f:
            for l in f:
                line = l.rsplit(" ",1)
                test_files.append(f"{line[0].strip()}")
                test_labs.append(classes[int(line[-1].strip())])
        test_data = list(zip(test_files, test_labs))

        return labeled_data, unlabeled_data, test_data

    elif dataset == "StanfordCars":
        labeled_files = []
        labels_files = []

        unlabeled_lab_files = []
        unlabeled_labs = []

        with open(f"{data_folder}/train.txt", "r") as f:
            for l in f:
                line = l.rsplit(" ",1)
                img = line[0].strip()
                cl = classes[int(line[-1].strip())]

                if cl in seen_classes:
                    labeled_files.append(f"{img}")
                    labels_files.append(cl)
                elif cl in unseen_classes:
                    unlabeled_lab_files.append(f"{img}")
                    unlabeled_labs.append(cl)
                else:
                    raise Exception(
                        f"The extracted class {cl} is not among the seen or unseen classes."
                    )
                
        labeled_data = list(zip(labeled_files, labels_files))
        unlabeled_data = list(zip(unlabeled_lab_files, unlabeled_labs))

        test_files = []
        test_labs = []

        with open(f"{data_folder}/test.txt", "r") as f:
            for l in f:
                line = l.rsplit(" ",1)
                test_files.append(f"{line[0].strip()}")
                test_labs.append(classes[int(line[-1].strip())])
        test_data = list(zip(test_files, test_labs))

        return labeled_data, unlabeled_data, test_data

    # Split labeled and unlabeled data into test
    train_labeled_files, train_labeles, test_seen_files, test_seen_labs = split_data(
        0.8, labeled_files, labels_files
    )
    labeled_data = list(zip(train_labeled_files, train_labeles))

    (
        train_unlabeled_files,
        train_un_labeles,
        test_unseen_files,
        test_unseen_labs,
    ) = split_data(0.8, unlabeled_lab_files, unlabeled_labs)
    unlabeled_data = list(zip(train_unlabeled_files, train_un_labeles))

    test_seen = list(zip(test_seen_files, test_seen_labs))
    test_unseen = list(zip(test_unseen_files, test_unseen_labs))

    test_data = test_seen + test_unseen

    return labeled_data, unlabeled_data, test_data


def split_data(ratio, files, labels):
    np.random.seed(500)
    train_indices = np.random.choice(
        range(len(files)), size=int(len(files) * ratio), replace=False
    )
    val_indices = list(set(range(len(files))).difference(set(train_indices)))

    train_labeled_files = np.array(files)[train_indices]
    train_labeles = np.array(labels)[train_indices]

    val_labeled_files = np.array(files)[val_indices]
    val_labeles = np.array(labels)[val_indices]

    return train_labeled_files, train_labeles, val_labeled_files, val_labeles
