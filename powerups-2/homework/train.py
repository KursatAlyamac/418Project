from planner import Planner, save_model 
import torch
import torch.utils.tensorboard as tb
import numpy as np
from utils import load_data
import dense_transforms
import matplotlib.pyplot as plt
import time

def train(args):
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    from os import path
    model = Planner()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

    """
    Your code here, modify your HW4 code
    
    """
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
    print("device:", device)
    model = model.to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'planner.th')))

    loss_path = torch.nn.L1Loss()
    loss_powerup = torch.nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), 
                                lr=args.learning_rate,
                                weight_decay=0.01)
    
    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})

    train_data = load_data('drive_data', transform=transform, num_workers=args.num_workers)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        epochs=args.num_epoch,
        steps_per_epoch=len(train_data),
        pct_start=0.3
    )

    global_step = 0
    with open('training_losses.txt', 'w') as f:
        best_loss = float('inf')
        for epoch in range(args.num_epoch):
            epoch_start_time = time.time()
            model.train()
            losses = []
            for img, label in train_data:
                img, label = img.to(device), label.to(device)

                # Zero gradients
                optimizer.zero_grad()
                
                # Get predictions
                aim_point, powerup_pred = model(img, return_powerup=True)
                
                # Compute losses with dynamic weighting
                path_loss = loss_path(aim_point, label)
                total_loss = path_loss
                
                if hasattr(label, 'powerup'):
                    powerup_loss = loss_powerup(powerup_pred, label.powerup)
                    # Dynamic weighting based on training progress
                    powerup_weight = min(0.3, 0.1 + epoch * 0.01)
                    total_loss += powerup_weight * powerup_loss
                
                # Backward pass with gradient clipping
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                if train_logger is not None:
                    train_logger.add_scalar('loss/total', total_loss.item(), global_step)
                    train_logger.add_scalar('loss/path', path_loss.item(), global_step)
                    if hasattr(label, 'powerup'):
                        train_logger.add_scalar('loss/powerup', powerup_loss.item(), global_step)
                
                losses.append(total_loss.item())
                global_step += 1
            
            epoch_time = time.time() - epoch_start_time
            avg_loss = np.mean(losses)
            
            status = 'epoch %-3d \t loss = %0.3f \t time = %0.1f sec' % (epoch, avg_loss, epoch_time)
            if train_logger is None:
                print(status)
            f.write(status + '\n')
            
            if train_logger is not None:
                train_logger.add_scalar('epoch_time', epoch_time, epoch)
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_model(model)

    save_model(model)

def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predited aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)])/2
    ax.add_artist(plt.Circle(WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=30)
    parser.add_argument('-w', '--num_workers', type=int, default=4)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform', default='Compose([ColorJitter(0.2, 0.5, 0.5, 0.2), RandomHorizontalFlip(), ToTensor()])')

    args = parser.parse_args()
    train(args)

    epochs = []
    losses = []
    times = []

    with open('training_losses.txt', 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            epoch = int(parts[0].split()[-1])
            loss = float(parts[1].split()[-1])
            time_str = parts[2].split('=')[-1].strip()
            time = float(time_str.replace('sec', '').strip())
            epochs.append(epoch)
            losses.append(loss)
            times.append(time)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    ax1.plot(epochs, losses, 'b-', label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(epochs, times, 'r-', label='Epoch Duration')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Training Time per Epoch')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

