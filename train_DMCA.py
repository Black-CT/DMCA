import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from torch_geometric.loader import DataLoader
import torch
import util as sd
from In_memory_dataset_whole_SMILE import MyOwnDataset
from model.multi_modality_SMILES import Net
import torch.utils.data
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
import pandas as pd


def model_eval(args, model, device, loader):
    number_of_task = args.n_tasks
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
            data.y = data.y.view(len(data), number_of_task)
            eval_loss += criterion(outputs, data.y)

            preds += outputs.view(-1).cpu().tolist()
            labels +=  data.y.cpu().tolist()


    trues = np.array(labels).reshape(-1, number_of_task).T
    belief_scores = np.array(preds).reshape(-1, number_of_task).T
    roc_auc_score_list = []
    for i in range(number_of_task):
        temp_roc_auc_score = roc_auc_score(trues[i].tolist(), belief_scores[i].tolist())
        roc_auc_score_list.append(temp_roc_auc_score)

    auc_roc = sum(roc_auc_score_list) / number_of_task
    accuracy = accuracy_score(labels, [1 if value > 0.5 else 0 for value in preds])

    return eval_loss, accuracy, auc_roc


if __name__ == '__main__':
    """ parameters """
    args = sd.parse_input()

    epochs = args.epochs
    train_batch_size = args.batch_size
    number_of_task = args.n_tasks
    training_task = args.dataset
    model_name = args.model_name
    random_seed = args.random_seed
    args.model_name = "DMCA"
    random_scaffold = args.random_scaffold
    is_balance = False

    tensorboard_log_dir = "log_scaffold/" + sd.def_log_dir(args)
    writer = SummaryWriter(tensorboard_log_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

    """ split data """
    our_dataset = MyOwnDataset(root="drug_data/")

    if random_scaffold:
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_loader, valid_loader, test_loader, train_size, valid_size, test_size =\
            sd.random_scaffold_split(our_dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.random_seed, batch_size=args.batch_size)
    else:
        train_loader, test_loader, train_size, test_size = sd.split_data(our_dataset, args.random_seed, args.batch_size, is_balance)


    # print data size to check
    print(train_size + test_size + valid_size)

    """ load model """
    model = Net(number_of_task).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    criterion = torch.nn.BCELoss()

    max_auc_roc = 0
    valid_loss_list = []
    valid_accuracy_list = []
    valid_auc_roc_list = []
    test_loss_list = []
    test_accuracy_list = []
    test_auc_roc_list = []

    for epoch in range(epochs):
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n_correct = 0
        train_n_label = 0
        preds = []
        labels = []

        model.train()
        for i, (data) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            smiles = data.smiles
            if args.token_length_smile == 0:
                inputs = tokenizer.batch_encode_plus(smiles, truncation=True, padding=True, return_tensors="pt")
            else:
                inputs = tokenizer.batch_encode_plus(smiles, max_length=args.token_length_smile, truncation=True,
                                                                pad_to_max_length=True, return_tensors="pt")
            inputs = inputs.to(device)
            outputs = model(data, inputs)

            data.y = data.y.view(len(data), number_of_task)
            data.y = data.y[:, 0].view(-1, 1)

            # gradient
            loss = criterion(outputs, data.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss

            preds += outputs.view(-1).cpu().tolist()
            labels += data.y.cpu().tolist()


        trues = np.array(labels).reshape(-1, number_of_task).T
        belief_scores = np.array(preds).reshape(-1, number_of_task).T
        roc_auc_score_list = []
        for i in range(number_of_task):
            temp_roc_auc_score = roc_auc_score(trues[i].tolist(), belief_scores[i].tolist())
            roc_auc_score_list.append(temp_roc_auc_score)

        auc_roc = sum(roc_auc_score_list) / number_of_task
        train_accuracy = accuracy_score(labels, [1 if value > 0.5 else 0 for value in preds])

        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/Acc', train_accuracy, epoch)
        writer.add_scalar('Train/AUC-ROC', auc_roc, epoch)



        """ evaluate """
        eval_loss = 0
        eval_acc = 0
        model.eval()



        valid_loss, valid_accuracy, valid_auc_roc = model_eval(args, model, device, valid_loader)
        test_loss, test_accuracy, test_auc_roc = model_eval(args, model, device, test_loader)


        writer.add_scalar('Valid/Loss', valid_loss, epoch)
        writer.add_scalar('Valid/Acc', valid_accuracy, epoch)
        writer.add_scalar('Valid/AUC_ROC', valid_auc_roc, epoch)
        writer.add_scalar('Test/Loss', test_loss, epoch)
        writer.add_scalar('Test/Acc', test_accuracy, epoch)
        writer.add_scalar('Test/AUC_ROC', test_auc_roc, epoch)

        valid_loss_list.append(valid_loss)
        valid_accuracy_list.append(valid_accuracy)
        valid_auc_roc_list.append(valid_auc_roc)
        test_loss_list.append(test_loss)
        test_accuracy_list.append(test_accuracy)
        test_auc_roc_list.append(test_auc_roc)


        stop_time = time.time()
        print("time is:{:.4f}s".format(stop_time-start_time))

    # Find the best accuracy and AUC ROC values
    max_index_in_valid = valid_auc_roc_list.index(max(valid_auc_roc_list))

    best_valid_accuracy = valid_accuracy_list[max_index_in_valid]
    best_valid_auc_roc = valid_auc_roc_list[max_index_in_valid]
    best_test_accuracy = test_accuracy_list[max_index_in_valid]
    best_test_auc_roc = test_auc_roc_list[max_index_in_valid]

    # Create a dictionary to store results with the seed as the row index
    data = {
        "Seed": [args.random_seed],
        "Best Validation Accuracy": [best_valid_accuracy],
        "Best Validation AUC ROC": [best_valid_auc_roc],
        "Best Test Accuracy": [best_test_accuracy],
        "Best Test AUC ROC": [best_test_auc_roc]
    }

    # Convert to a DataFrame
    df = pd.DataFrame(data)
    df.set_index("Seed", inplace=True)  # Use the random seed as the index

    # Define CSV file name
    csv_file = args.key + "_key_" + args.dataset + "_best_metrics.csv"

    # Append to the CSV if it exists; otherwise, create a new one
    if os.path.exists(csv_file):
        existing_df = pd.read_csv(csv_file, index_col="Seed")
        combined_df = pd.concat([existing_df, df])  # Append the new row
        combined_df.to_csv(csv_file)
    else:
        df.to_csv(csv_file)

    # trues = trues[0]
    # image = sd.plot_confusion_matrix_image(confusion_matrix_preds, trues, ["Class 0", "Class 1"])
    # print(args.model_name, train_batch_size, args.random_seed, max_auc_roc)

    # # Log the confusion matrix image to TensorBoard
    # writer.add_image("Confusion Matrix", image.permute(2, 0, 1))
    # writer.close()
    # # sd.save_model(epoch, model_state_dict, optimizer_state_dict, best_loss, tensorboard_log_dir)



