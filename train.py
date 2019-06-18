import torch
import torchvision.transforms as transforms
import transforms as ext_transforms
import utils
from collections import OrderedDict


class Train:
    """Performs the training of ``model`` given a training dataset data
    loader, the optimizer, and the loss criterion.

    Keyword arguments:
    - model (``nn.Module``): the model instance to train.
    - data_loader (``Dataloader``): Provides single or multi-process
    iterators over the dataset.
    - optim (``Optimizer``): The optimization algorithm.
    - criterion (``Optimizer``): The loss criterion.
    - metric (```Metric``): An instance specifying the metric to return.
    - device (``torch.device``): An object representing the device on which
    tensors are allocated.

    """

    

    def __init__(self, model, data_loader, optim, criterion, metric, device):
        self.model = model
        self.data_loader = data_loader
        self.optim = optim
        self.criterion = criterion
        self.metric = metric
        self.device = device
        self.color_encoding = OrderedDict([
            ('unlabeled', (0, 0, 0)),
            ('road', (128, 64, 128)),
            ('sidewalk', (244, 35, 232)),
            ('building', (70, 70, 70)),
            ('wall', (102, 102, 156)),
            ('fence', (190, 153, 153)),
            ('pole', (153, 153, 153)),
            ('traffic_light', (250, 170, 30)),
            ('traffic_sign', (220, 220, 0)),
            ('vegetation', (107, 142, 35)),
            ('terrain', (152, 251, 152)),
            ('sky', (70, 130, 180)),
            ('person', (220, 20, 60)),
            ('rider', (255, 0, 0)),
            ('car', (0, 0, 142)),
            ('truck', (0, 0, 70)),
            ('bus', (0, 60, 100)),
            ('train', (0, 80, 100)),
            ('motorcycle', (0, 0, 230)),
            ('bicycle', (119, 11, 32))
            ])

    def run_epoch(self, epoch, iteration_loss=False):
        """Runs an epoch of training.

        Keyword arguments:
        - iteration_loss (``bool``, optional): Prints loss at every step.

        Returns:
        - The epoch loss (float).

        """
        self.model.train()
        epoch_loss = 0.0
        self.metric.reset()
        for step, batch_data in enumerate(self.data_loader):

            if step > 0:
                break

            # Get the inputs and labels
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)
            #print('labels!!!!!!!!!!!!!!!!!!!!!!',labels)

            # Forward propagation
            outputs = self.model(inputs)
            _, preds = torch.max(outputs.data, 1)

            # Loss computation
            loss = self.criterion(outputs, labels)

            # Backpropagation
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # Keep track of loss for current epoch
            epoch_loss += loss.item()

            # Keep track of the evaluation metric
            self.metric.add(outputs.detach(), labels.detach())

            if iteration_loss:
                print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))
            
            if epoch == 59 or epoch == 99 or epoch == 159 or epoch == 199:
                if step % 500 == 0:
                    print('Visualization of Train:')
                    label_to_rgb = transforms.Compose([
                    ext_transforms.LongTensorToRGBPIL(self.color_encoding),
                    transforms.ToTensor()
                    ])
                    color_labels = utils.batch_transform(labels.cpu(), label_to_rgb)
                    color_outputs = utils.batch_transform(preds.cpu(), label_to_rgb)
                    utils.imshow_batch(color_outputs, color_labels)
                
            
            if epoch == 80:
                torch.save(self.model.state_dict(), '/home/wan/PyTorch-ENet/model_80')
            if epoch == 120:
                torch.save(self.model.state_dict(), '/home/wan/PyTorch-ENet/model_120')
            if epoch == 140:
                torch.save(self.model.state_dict(), '/home/wan/PyTorch-ENet/model_140')
            if epoch == 160:
                torch.save(self.model.state_dict(), '/home/wan/PyTorch-ENet/model_160')
            if epoch == 180:
                torch.save(self.model.state_dict(), '/home/wan/PyTorch-ENet/model_180')
            if epoch == 199:
                torch.save(self.model.state_dict(), '/home/wan/PyTorch-ENet/model_200')
            
        return epoch_loss / len(self.data_loader), self.metric.value()
