import torch
import torch.nn as nn
from DataLoader import DatasetLoader
import os
from custom_models.models import CustomModels
import torch.optim as optim
from config import get_config
from KD_Lib.KD import RCO
import glob
import sys

config = get_config()

dataset = sys.argv[1]
num_classes = 10
if dataset == 'fashion_mnist':
    in_channel = 1
elif dataset == 'cifar10':
    in_channel = 3

cmi = CustomModels(IN_CHANNEL=in_channel, NUM_OUTPUT=num_classes)
KD_METHOD = 'RCO'

# Teacher and Student Model Skeleton
teacher_models = [
    cmi.init_model('efficientnet-b7')
]

student_models = [
    cmi.init_model('model_25k_w_dw'),
    cmi.init_model('model_25k_wo_dw')
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
# RCO Doesnt Require valid dataset
dl = DatasetLoader(ds=dataset)
train_dl, test_dl = dl.getDataLoader(valid=False)
print("Dataset: ", str(dl._name))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)

# 1. RCO - Online Distillation
# Define student and teacher models
# Both Teacher and Student models are trained from scratch
# Num epochs for teacher and student  = 25, 25 anchod points along the route and incrementally distill to student.
# config.batch_size = 1024 # For RCO (I GOT GPU MEMORY..)
# Train Teacher every epoch -> Learn Anchor points and persist -> Train student model using the anchor points
n_anchor_points = 5
for teacher_model in teacher_models:

    # Train Teacher Model only once and get the anchor points
    teacher_model = teacher_model.to(device)
    teacher_optimizer = optim.Adam(teacher_model.parameters(), lr=config.learning_rate)

    distiller = None

    for student_model in student_models:
        
        print("----STARTED----")
        print(teacher_model._name, student_model._name)

        # Preload weights and optimizer state for teacher model
        # loss_fn = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(teacher_model.parameters(), lr=config.learning_rate)

        # DO NOT LOAD PRETRAINED TEACHER MODEL FOR RCO - It needs to learn anchor points alongside student model

        # ckpt_model_path = getCheckpointModelPath('Teacher', teacher_model._name)

        # checkpoint = torch.load(ckpt_model_path)
        # teacher_model.load_state_dict(checkpoint['model_state_dict'])
        # teacher_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']

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
            print("Distiller Initialized")
            distiller = RCO(teacher_model=teacher_model,
                            student_model=student_model,
                            train_loader=train_dl,
                            val_loader=test_dl,
                            optimizer_teacher=teacher_optimizer,
                            optimizer_student=student_optimizer,
                            epoch_interval=int(config.dist_train_epochs/n_anchor_points),
                            device=device,
                            log=True,
                            logdir=logdir)
            print("Teacher epoch_interval: ")
            print(distiller.epoch_interval)
            distiller.train_teacher(epochs=config.dist_train_epochs, save_model_pth=teacher_save_model_pth) # Remember to comment this. Train the teacher model
            distiller.epoch_interval = int(config.dist_val_epochs/n_anchor_points)
        else:
            distiller.student_model = student_model
            distiller.optimizer_student = student_optimizer
        
        print("Student epoch_interval:")
        print(distiller.epoch_interval)
        print("Distiller Teacher Model")
        print(distiller.teacher_model._name)
        print("Distiller Student Model")
        print(distiller.student_model._name)

        distiller.train_student(epochs=config.dist_val_epochs, save_model_pth=student_save_model_pth) # Train the student model
        distiller.evaluate(teacher=True) # Evaluate the teacher model
        distiller.evaluate()

        print(teacher_model._name, student_model._name)
        print("----COMPLETED----")
        