import argparse
import os
from datetime import datetime
from io import BytesIO
from torch_geometric.loader import DataLoader
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
from itertools import compress
from collections import defaultdict
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm


def def_log_dir(args) -> str:
    if args.model_name == "smile_chembert":
        if args.token_length_smile == 0:
            args.model_name = args.model_name + "_unlimited_length"
        else:
            args.model_name = args.model_name + "_{}_length".format(args.token_length_smile)
    if args.model_name == "own_iupac_pretraining":
        if args.token_length_iupac == 0:
            args.model_name = args.model_name + "_unlimited_length"
        else:
            args.model_name = args.model_name + "_{}_length".format(args.token_length_iupac)

    if args.model_name == "fusion_smile_and_iupac":
        args.model_name = "fusion_smile_{}_and_iupac_{}".format(args.token_length_smile, args.token_length_iupac)

    print(args.model_name)

    tensorboard_log_dir = "log/" + args.model_name + "/" + args.dataset + "/epoch_" + str(args.epochs) + \
                          "/batchsize_" + str(args.batch_size) +\
                          "/random_seed" + str(args.random_seed) + \
                          "/key" + str(args.key) + \
                          "/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    return tensorboard_log_dir


def model_eval(args, model, device, loader, tokenizer, criterion):
    eval_loss = 0
    accuracy = 0
    auc_roc = 0
    preds = []
    labels = []
    for i, data in enumerate(tqdm(loader)):
        with torch.no_grad():
            data = data.to(device)
            smiles = data.smiles
            if args.token_length_smile == 0:
                inputs = tokenizer.batch_encode_plus(smiles, truncation=True, padding=True, return_tensors="pt")
            else:
                inputs = tokenizer.batch_encode_plus(smiles, max_length=args.token_length_smile,
                                                     truncation=True, pad_to_max_length=True, return_tensors="pt")
            inputs = inputs.to(device)
            outputs = model(data, inputs)
            data.y = data.y.view(len(data), args.n_tasks)
            # data.y = data.y[:, 0].view(-1, 1)
            eval_loss += criterion(outputs, data.y)

            preds += outputs.view(-1).cpu().tolist()
            labels += data.y.cpu().tolist()

    trues = np.array(labels).reshape(-1, args.n_tasks).T
    belief_scores = np.array(preds).reshape(-1, args.n_tasks).T
    roc_auc_score_list = []
    for i in range(args.n_tasks):
        temp_roc_auc_score = roc_auc_score(trues[i].tolist(), belief_scores[i].tolist())
        roc_auc_score_list.append(temp_roc_auc_score)

    auc_roc = sum(roc_auc_score_list) / args.n_tasks
    accuracy = accuracy_score(labels, [1 if value > 0.5 else 0 for value in preds])

    return eval_loss, accuracy, auc_roc

# execution command
# python train_iupac_own_tokenizer.py --dataset "BBBP" --batch_size 64 --epochs 200 --n_tasks 1 --model_name own_iupac_pretraining
# python train_whole_SMILE.py --dataset "BBBP" --batch_size 64 --epochs 200 --n_tasks 1 --model_name smile_chembert
# python train_SMILE_and_IUPAC.py --dataset "BBBP" --batch_size 64 --epochs 200 --n_tasks 1 --model_name fusion_smile_and_iupac
def parse_input():
    """ parameters """
    parser = argparse.ArgumentParser(description='prediction based on Multimodality')
    parser.add_argument('--dataset', type=str, default='Lipophilicity', help='Name of dataset')
    parser.add_argument('--model_name', type=str, default="", help='the root directory of the log file,smile_chembert, own_iupac_pretraining, fusion_smile_and_iupac')
    parser.add_argument('--n_tasks', type=int, default=1, help='Number of label')
    parser.add_argument('--random_seed', type=int, default=0, help='random_seed')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--token_length_smile', type=int, default=0, help='token length of smile')
    parser.add_argument('--token_length_iupac', type=int, default=0, help='token length of iupac')
    parser.add_argument('--key', type=str, default="smile", help='key of embedding')
    parser.add_argument('--random_scaffold', type=bool, default="True", help='key of embedding')


    # parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')


    args = parser.parse_args()
    return args


def plot_confusion_matrix_image(y_pred, y_true, class_labels=["Class 0", "Class 1"]):
    print(y_true)
    print(y_pred)
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)

    # Plot the confusion matrix
    fig, ax = plt.subplots()
    disp.plot(cmap="Blues", ax=ax, values_format=".0f")

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # Convert the BytesIO object to a PIL Image and then to a torch tensor
    image = Image.open(buf)
    image = torch.tensor(np.array(image))
    return image

def save_model(best_epoch, model_state_dict, optimizer_state_dict, best_loss, log_directory):
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'loss': best_loss,
    }, os.path.join(log_directory, 'checkpoint.pth'))


def split_data(our_dataset, random_seed, train_batch_size, is_balance=False):
    print("size of dataset", len(our_dataset))

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

    setup_seed(random_seed)

    train_size = int(0.8 * len(our_dataset))
    test_size = len(our_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(our_dataset, [train_size, test_size])

    #  balanced training
    if is_balance:
        n_y_positive = 0
        n_y_negative = 0
        idx_balanced_train_dataset = []

        for idx in train_dataset.indices:
            if our_dataset[idx].y == 1:
                n_y_positive += 1
                idx_balanced_train_dataset.append(idx)

        for idx in train_dataset.indices:
            if our_dataset[idx].y == 0:
                n_y_negative += 1
                idx_balanced_train_dataset.append(idx)
                if n_y_negative == n_y_positive:
                    break

        balanced_train_dataset = torch.utils.data.Subset(our_dataset, idx_balanced_train_dataset)
        print("the number of balanced training dataset ", len(balanced_train_dataset))
        train_dataset = balanced_train_dataset

    """ data loader """
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size,shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=train_batch_size,shuffle=True)

    return train_loader, test_loader, train_size, test_size



def add_regression_data_number_scaler(writer, number):
    writer.add_scalar('Number of testing data', number)

def compute_std_mean(result):
    nums = np.array(result)
    std = np.std(np.array(nums))
    mean = np.mean(nums)
    print("reuslt: {} | {:.4} + {:.3}".format(nums, mean, std))



def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold


def random_scaffold_split(dataset, smiles_list, task_idx=None, null_value=0,
                   frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0, batch_size=16):
    """
    Adapted from https://github.com/pfnet-research/chainer-chemistry/blob/master/chainer_chemistry/dataset/splitters/scaffold_splitter.py
    Split dataset by Bemis-Murcko scaffolds
    This function can also ignore examples containing null values for a
    selected task when splitting. Deterministic split
    :param dataset: pytorch geometric dataset obj
    :param smiles_list: list of smiles corresponding to the dataset obj
    :param task_idx: column idx of the data.y tensor. Will filter out
    examples with null value in specified task column of the data.y tensor
    prior to splitting. If None, then no filtering
    :param null_value: float that specifies null value in data.y to filter if
    task_idx is provided
    :param frac_train:
    :param frac_valid:
    :param frac_test:
    :param seed;
    :return: train, valid, test slices of the input dataset obj
    """

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
    setup_seed(seed)
    rng = np.random.RandomState(seed)

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if task_idx != None:
        # filter based on null values in task_idx
        # get task array
        y_task = np.array([data.y[task_idx].item() for data in dataset])
        # boolean array that correspond to non null values
        non_null = y_task != null_value
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    else:
        non_null = np.ones(len(dataset)) == 1
        smiles_list = list(compress(enumerate(smiles_list), non_null))



    scaffolds = defaultdict(list)
    invalid_smiles = []
    for ind, smiles in smiles_list:
        try:
            scaffold = generate_scaffold(smiles, include_chirality=True)
            scaffolds[scaffold].append(ind)
        except ValueError as e:
            invalid_smiles.append(smiles)
    print("invalid smiles \n", invalid_smiles)



    # random
    scaffold_sets = rng.permutation(np.array(list(scaffolds.values()), dtype=object))

    n_total_valid = int(np.floor(frac_valid * len(dataset)))
    n_total_test = int(np.floor(frac_test * len(dataset)))

    train_idx = []
    valid_idx = []
    test_idx = []

    for scaffold_set in scaffold_sets:
        if len(valid_idx) + len(scaffold_set) <= n_total_valid:
            valid_idx.extend(scaffold_set)
        elif len(test_idx) + len(scaffold_set) <= n_total_test:
            test_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)


    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(valid_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, len(train_idx), len(valid_idx), len(test_idx)

# from rdkit import Chem
# from rdkit.Chem.Scaffolds import MurckoScaffold
#
# # Original SMILES
# smiles = 'O=N([O-])C1=C(CN=C1NCCSCc2ncccc2)Cc3ccccc3'
# mol = Chem.MolFromSmiles(smiles)
# print(mol)
