
"""Trainer for DeepPhys."""
import os

import numpy as np
import torch
import torch.optim as optim
from neural_methods.model.DeepPhys import DeepPhys
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm


class CustomTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0
        
        if config.TOOLBOX_MODE == "only_test":
            self.model = DeepPhys(img_size=config.TEST.DATA.PREPROCESS.RESIZE.H).to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
        else:
            raise ValueError("Custom trainer only supports 'only_test' as a TOOLBOX_MODE")

    def train(self, data_loader):
        raise NotImplementedError("Custom trainer doesn't impelement a training loop")

    def valid(self, data_loader):
        """ Model evaluation on the validation dataset."""
        raise NotImplementedError("Custom trainer doesn't impelement a validation loop")

    def test(self, data_loader):
        """ Model evaluation on the testing dataset."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print("===Testing===")
        predictions = dict()
        labels = dict()
        if self.config.TOOLBOX_MODE != "only_test":
            raise ValueError("Custom trainer only supports 'only_test' as a TOOLBOX_MODE")
        if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
            raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
        self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH, map_location=torch.device("cpu")))
        print("Testing uses pretrained model!")

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                data_test, labels_test = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                N, D, C, H, W = data_test.shape
                data_test = data_test.view(N * D, C, H, W)
                labels_test = labels_test.view(-1, 1)
                pred_ppg_test = self.model(data_test)

                if self.config.TEST.OUTPUT_SAVE_DIR:
                    labels_test = labels_test.cpu()
                    pred_ppg_test = pred_ppg_test.cpu()

                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels[subj_index][sort_index] = labels_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
        
        print('')
        for key, value in predictions.items():
            for key_1, value_1 in value.items():
                print(f"predictions of key {key_1} has shape: {value_1.shape}")
        # calculate_metrics(predictions, labels, self.config)
        if self.config.TEST.OUTPUT_SAVE_DIR: # saving test outputs
            self.save_test_outputs(predictions, labels, self.config)

    def save_model(self, index):
        """Inits parameters from args and the writer for TensorboardX."""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
 
