from custom_models.models import CustomModels
from DataLoader import DatasetLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
from helper import accuracy_fn
from config import get_config
from helper.utils import CheckpointBestModel
import torch.optim as optim
import os
import glob
import sys
config = get_config()

torch.manual_seed(42)
dataset = sys.argv[1]
num_classes = 10
if dataset == 'fashion_mnist':
    in_channel = 1
elif dataset == 'cifar10':
    in_channel = 3
    config.learning_rate = 0.01

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

cmi = CustomModels(IN_CHANNEL=in_channel, NUM_OUTPUT=num_classes) # (3, 10) CIFAR-10 & (1, 10) FashionMNIT and (3, 100) CIFAR 100

train_models = [
    cmi.init_model('model_25k_w_dw'),
    cmi.init_model('model_25k_wo_dw')
    # cmi.init_model('resnet_18'),
    # cmi.init_model('resnet_34'),
    # cmi.init_model('model_143k_w_dw'),
    # cmi.init_model('model_143k_wo_dw'),
    # cmi.init_model('model_340k_w_dw'),
    # cmi.init_model('model_340k_wo_dw'),
    # cmi.init_model('model_600k_w_dw'),
    # cmi.init_model('model_600k_wo_dw'),
    # cmi.init_model('model_1M_w_dw'),
    # cmi.init_model('model_1M_wo_dw')
    # cmi.init_model('resnet_50'),
    # cmi.init_model('resnet_101'),
    # cmi.init_model('resnet_152')
    # cmi.init_model('efficientnet-b5'),
    # cmi.init_model('efficientnet-b7')
]


dl = DatasetLoader(ds=dataset)
train_dl, valid_dl, test_dl = dl.getDataLoader(valid=True)
print("Dataset: ", str(dl._name))
dl._name = dl._name

def update_lr(opt, lr):    
    for param_group in opt.param_groups:
        param_group['lr'] = lr

def train(model, num_epochs, train_dl, valid_dl, loss_fn, optimizer):
    loss_hist_train = [] 
    accuracy_hist_train = [] 
    loss_hist_valid = []
    accuracy_hist_valid = []
    curr_lr = config.learning_rate

    checkpoint = CheckpointBestModel()
    for epoch in range(num_epochs):
        model.train()

        loss_hist_train.append(0)
        accuracy_hist_train.append(0) 
        loss_hist_valid.append(0) 
        accuracy_hist_valid.append(0)

        for x_batch, y_batch in train_dl:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            
            loss_hist_train[-1] += loss.item()
            accuracy_hist_train[-1] += accuracy_fn(y_true=y_batch, y_pred=pred.argmax(dim=1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        loss_hist_train[-1] /= len(train_dl.dataset)
        accuracy_hist_train[-1] /= len(train_dl.dataset)
    
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss_hist_valid[-1] += loss.item()
                accuracy_hist_valid[-1] += accuracy_fn(y_true=y_batch,y_pred=pred.argmax(dim=1))

        loss_hist_valid[-1] /= len(valid_dl.dataset)
        accuracy_hist_valid[-1] /= len(valid_dl.dataset)


        print(f'Epoch {epoch + 1} train_accuracy: '
              f'{accuracy_hist_train[epoch]:.4f} val_accuracy: '
              f'{accuracy_hist_valid[epoch]:.4f}')


        checkpoint(dataset_name=dl._name,
                   current_valid_loss=round(loss_hist_valid[-1], 4),
                   current_valid_accuracy=round(accuracy_hist_valid[-1], 3),
                   epoch=epoch,
                   model=model,
                   optimizer=optimizer,
                   criterion=loss_fn)
        
        # Decay learning rate
        if config.num_epochs >= 5 and (epoch+1) % int(config.num_epochs / 5) == 0:
            curr_lr /= 1.5
            update_lr(optimizer, curr_lr)
            print("Reduced Learning Rate:")
            print(optimizer)


    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid

def eval_model(model, test_dl, loss_fn, device):
    loss, acc = 0, 0
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in test_dl:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            loss += loss_fn(pred, y_batch)
            acc += accuracy_fn(y_true=y_batch, y_pred=pred.argmax(dim=1))

        loss /= len(test_dl.dataset)
        acc /= len(test_dl.dataset)

    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}


for idx, model in enumerate(train_models):

    print(f"Starting {idx}. {model._name}")
    model = model.to(device)

    # Loss function and Optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 
                            lr=config.learning_rate)

    # Model Train and Validation loop
    hist = train(model=model, 
                    num_epochs=config.num_epochs,
                    train_dl=train_dl, 
                    valid_dl=valid_dl,
                    loss_fn=loss_fn,
                    optimizer=optimizer)
    print(hist)

    save_path = os.path.join(config.ckpt_path,
                            dl._name,
                            model._name)

    best_model_path = os.path.join(max(glob.glob(os.path.join(save_path, '*/')), key=os.path.getmtime), 'best_model.pth')
    print("Best Model Path: ")
    print(best_model_path)

    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_hist = eval_model(model=model, 
                            test_dl=test_dl,
                            device=device,
                            loss_fn=loss_fn)
    print(test_hist)

    print(f"Completed {idx}. {model._name}")

# # Plot
# #%%
# import numpy as np
# import matplotlib.pyplot as plt

# x_arr = np.arange(len(hist[0])) + 1
# fig = plt.figure(figsize=(12, 4))
# ax = fig.add_subplot(1, 2, 1)
# ax.plot(x_arr, hist[0], '-o', label='Train Loss')
# ax.plot(x_arr, hist[1], '--<', label='Validation Loss')
# ax.legend(fontsize=15)

# ax = fig.add_subplot(1, 2, 2)
# ax.plot(x_arr, hist[2], '-o', label='Train Accuracy')
# ax.plot(x_arr, hist[3], '--<', label='Validation Accuracy')
# ax.legend(fontsize=15)

# ax.set_xlabel('Epoch', size=15)
# ax.set_ylabel('Accuracy', size=15)
# plt.show()
# # %%