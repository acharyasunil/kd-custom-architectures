import torch
import torch.nn as nn
from DataLoader import DatasetLoader
import os
from custom_models.models import CustomModels
import torch.optim as optim
from config import get_config
from KD_Lib.KD import SelfTraining
import glob
import sys
config = get_config()

config.batch_size = 2048
config.dist_val_epochs = 70

dataset = sys.argv[1]
num_classes = 10
if dataset == 'fashion_mnist':
    in_channel = 1
elif dataset == 'cifar10':
    in_channel = 3

# Implements SelfTraining Method, Teacherless

cmi = CustomModels(IN_CHANNEL=in_channel, NUM_OUTPUT=num_classes) # (3, 10) CIFAR-10 & (1, 10) FashionMNIT and (3, 100) CIFAR 100
KD_METHOD = 'TF_ST'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# 4. TEACHER_FREE - No Teacher Model. Only Student Model
student_models = [
    cmi.init_model('model_25k_w_dw'),
    cmi.init_model('model_25k_wo_dw')
    # cmi.init_model('resnet_18'),
    # cmi.init_model('resnet_34'),
    # cmi.init_model('resnet_50'),
    # cmi.init_model('resnet_101'),
    # cmi.init_model('efficientnet-b5'),
    # cmi.init_model('efficientnet-b7')
    # cmi.init_model('model_143k_w_dw'),
    # cmi.init_model('model_143k_wo_dw'),
    # cmi.init_model('model_340k_w_dw'),
    # cmi.init_model('model_340k_wo_dw'),
    # cmi.init_model('model_600k_w_dw'),
    # cmi.init_model('model_600k_wo_dw'),
    # cmi.init_model('model_1M_w_dw'),
    # cmi.init_model('model_1M_wo_dw')
]


def getCheckpointModelPath(model_base_dir, dataset_name, model_type):
    model_path = os.path.join(model_base_dir, dataset_name, model_type)
    model_path = os.path.abspath(glob.glob(f'{model_path}/*/*.pth')[0])
    return model_path

# Get the dataset loader
dl = DatasetLoader(ds=dataset)
train_dl, test_dl = dl.getDataLoader(valid=False)
print("Dataset: ", str(dl._name))

# 4. Teacher Free - Self Training
# Only Student Model.
# Batch - 1024
for student_model in student_models:
    
    print("----STARTED TF_ST----")
    print(student_model._name)

    # Fresh out of the oven, student models
    student_model = student_model.to(device)
    student_optimizer = optim.Adam(student_model.parameters(), lr=config.learning_rate)

    # KD Teacher save model path
    #teacher_save_model_pth = os.path.join('kd_models_save', KD_METHOD, '_'+ student_model._name, 'teacher.pth')
    student_save_model_pth = os.path.join('kd_models_save', dl._name, KD_METHOD, '_'+ student_model._name, 'student.pth')
    dir_name = os.path.dirname(student_save_model_pth)
    os.makedirs(dir_name, exist_ok=True)

    # Experiment Tensorboard Log Directory
    logdir = os.path.join('./Experiments', dl._name, KD_METHOD)
    os.makedirs(logdir, exist_ok=True)


    print("Teacher-Free Self Training Initialized")
    distiller = SelfTraining(student_model=student_model,
                    train_loader=train_dl,
                    val_loader=test_dl,
                    optimizer_student=student_optimizer,
                    device=device,
                    log=True,
                    logdir=logdir)
    
    distiller.train_student(epochs=config.dist_val_epochs, save_model_pth=os.path.join(dir_name, 'student.pth')) # Train the student model
    distiller.evaluate()

    print(student_model._name)
    print("----COMPLETED----")
        