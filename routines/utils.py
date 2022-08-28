from webbrowser import get
import torch
import matplotlib.pyplot as plt
from config import get_config
config = get_config()


plt.style.use('ggplot')
import os

class CheckpointBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, dataset_name, current_valid_loss, current_valid_accuracy,
        epoch, model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            save_path = os.path.join(config.ckpt_path,
                                    dataset_name,
                                    model._name,
                                    f'ep_{epoch + 1}_vacc_{current_valid_accuracy}_vloss_{current_valid_loss}', 
                                    'best_model.pth')

            dir_path = os.path.dirname(os.path.realpath(save_path))
            os.makedirs(dir_path, exist_ok = True)

            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, save_path)