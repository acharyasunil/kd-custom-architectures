from ntpath import join
import torch
import torch.nn as nn
from DataLoader import DatasetLoader
import os
from custom_models.models import CustomModels
import torch.optim as optim
from config import get_config
from KD_Lib.KD import TAKD
import glob
import sys
config = get_config()

KD_METHOD = 'TAKD'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)

dataset = sys.argv[1]
num_classes = 10
if dataset == 'fashion_mnist':
    in_channel = 1
elif dataset == 'cifar10':
    in_channel = 3

cmi = CustomModels(IN_CHANNEL=in_channel, NUM_OUTPUT=num_classes)


# Teacher and Student Model Skeleton
teacher_models = [
    cmi.init_model('efficientnet-b7')
    # cmi.init_model('resnet_152')
]

student_models = [
    cmi.init_model('model_25k_w_dw'),
    cmi.init_model('model_25k_wo_dw'),
    # cmi.init_model('model_143k_w_dw'),
    # cmi.init_model('model_143k_wo_dw'),
    # cmi.init_model('model_340k_w_dw'),
    # cmi.init_model('model_340k_wo_dw'),
    # cmi.init_model('model_600k_w_dw'),
    # cmi.init_model('model_600k_wo_dw'),
    # cmi.init_model('model_1M_w_dw'),
    # cmi.init_model('model_1M_wo_dw')
]

teacher_assistants = [
    # cmi.init_model('resnet_18').to(device),
    cmi.init_model('resnet_34').to(device),
    cmi.init_model('efficientnet-b5').to(device)
]
optimizer_assistants = [
    optim.Adam(teacher_assistants[0].parameters(), lr=config.learning_rate),
    optim.Adam(teacher_assistants[1].parameters(), lr=config.learning_rate)
]

def getCheckpointModelPath(model_base_dir, dataset_name, model_type):
    model_path = os.path.join(model_base_dir, dataset_name, model_type)
    model_path = os.path.abspath(glob.glob(f'{model_path}/*/*.pth')[0])
    return model_path

dl = DatasetLoader(ds=dataset)
train_dl, test_dl = dl.getDataLoader(valid=False)
print("Dataset: ", str(dl._name))

# 2. TAKD - Offline Distillation
# Pre-Trained Teacher - No need to train again. Just used the checkpoint model and restore
n_anchor_points = 5
for teacher_model in teacher_models:

    # Train Teacher Model only once and get the anchor points
    teacher_model = teacher_model.to(device)
    teacher_optimizer = optim.Adam(teacher_model.parameters(), lr=config.learning_rate)

    ckpt_model_path = getCheckpointModelPath('Teacher', dl._name, teacher_model._name)

    checkpoint = torch.load(ckpt_model_path)
    teacher_model.load_state_dict(checkpoint['model_state_dict'])
    teacher_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']

    distiller = None

    for student_model in student_models:
        
        print("----STARTED----")
        print(teacher_model._name, student_model._name)

        # Fresh out of the oven, student models
        student_model = student_model.to(device)
        student_optimizer = optim.Adam(student_model.parameters(), lr=config.learning_rate)

        # KD Teacher save model path
        teacher_save_model_pth = os.path.join('kd_models_save', dl._name, KD_METHOD, teacher_model._name +'_'+ student_model._name, 'teacher.pth')
        student_save_model_pth = os.path.join('kd_models_save', dl._name, KD_METHOD, teacher_model._name +'_'+ student_model._name, 'student.pth')
        dir_name = os.path.dirname(teacher_save_model_pth)
        os.makedirs(dir_name, exist_ok=True)

        # Experiment Tensorboard Log Directory
        logdir = os.path.join('./Experiments', dl._name, KD_METHOD)
        os.makedirs(logdir, exist_ok=True)

        # KD Method Training
        # Initialize distiller only once and train teacher model only once
        # 50 epochs / 10 anchor points - 5 interval
        if distiller is None:
            print("TAKD Initialized")
            distiller = TAKD(teacher_model=teacher_model,
                            assistant_models=teacher_assistants,
                            student_model=student_model,
                            assistant_train_order=[[-1], [-1]],
                            train_loader=train_dl,
                            val_loader=test_dl,
                            optimizer_teacher=teacher_optimizer,
                            optimizer_assistants=optimizer_assistants,
                            optimizer_student=student_optimizer,
                            device=device,
                            log=True,
                            logdir=logdir)
            distiller.train_assistants(epochs=config.dist_train_epochs, save_dir=os.path.join(dir_name, '')) # Remember to comment this. Train the teacher model
        else:
            distiller.student_model = student_model
            distiller.optimizer_student = student_optimizer

        print("Distiller Teacher Model")
        print(distiller.teacher_model._name)
        print("Distiller Student Model")
        print(distiller.student_model._name)

        distiller.train_student(epochs=config.dist_val_epochs, save_model_pth=os.path.join(dir_name, 'student.pt')) # Train the student model
        distiller.evaluate(teacher=True) # Evaluate the teacher model
        distiller.evaluate()

        print(teacher_model._name, student_model._name)
        print("----COMPLETED----")
        