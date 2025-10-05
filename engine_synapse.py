import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
import torch

from sklearn.metrics import confusion_matrix
from scipy.ndimage.morphology import binary_fill_holes, binary_opening
from utils import test_single_volume
import time
from boundary.utils import *
from boundary.losses import *
from boundary.boundary_labels import *


def train_one_epoch(train_loader,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    epoch,
                    logger,
                    config,
                    writer,
                    scaler=None):
    '''
    train model for one epoch
    '''
    stime = time.time()
    model.train()

    loss_list = []

    initial_alpha = 0.01
    alpha_step = 0.0003
    α = initial_alpha + epoch * alpha_step

    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()

        images, image_t1s, targets = data['image'], data['image_t1'], data['label']
        images, image_t1s, targets = images.cuda(non_blocking=True).float(), image_t1s.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

        if config.amp:
            with autocast():
                out = model(images)
                loss = criterion(out, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(images)

            CeDice_loss = criterion(out, targets)

            boundary_loss = BoundaryLoss(idc=[3])
            device = 'cuda'

            targets_bl = targets.cpu().numpy().astype(int)
            targets_bl = np_class2one_hot(targets_bl, out.shape[1])

            dis = batch_one_hot2dist(targets_bl, [1.0, 1.0])
            dis = torch.tensor(dis).to(device)
            out = F.softmax(out, dim=1)
            bl_loss = boundary_loss(out, dis)  # Notice we do not give the same input to that loss

            boundary_labels = generate_boundary_labels(targets, num_classes=4)
            out_boundary = torch.sigmoid(out_boundary)
            boundary_labels = boundary_labels.float()
            criterion1 = nn.BCEWithLogitsLoss()


            loss_boundary = criterion1(out_boundary, boundary_labels)

            loss = CeDice_loss + 0.2 * bl_loss + 0.5 * loss_boundary


            loss.backward()
            optimizer.step()



        loss_list.append(loss.item())
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        mean_loss = np.mean(loss_list)

        writer.add_scalar('loss', loss, epoch)  # 我加的

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {loss.item():.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)

    scheduler.step()
    etime = time.time()
    log_info = f'Finish one epoch train: epoch {epoch}, loss: {mean_loss:.4f}, time(s): {etime - stime:.2f}'
    print(log_info)
    logger.info(log_info)
    return mean_loss


def val_one_epoch(test_datasets,
                  test_loader,
                  model,
                  epoch,
                  logger,
                  config,
                  test_save_path,
                  val_or_test=False):
    # switch to evaluate mode
    stime = time.time()
    model.eval()
    with torch.no_grad():
        metric_list = 0.0
        preds = []
        gts = []
        i_batch = 0
        for data in tqdm(test_loader):

            img, img_t1, msk, case_name = data['image'], data['image_t1'], data['label'], data['case_name'][0]
            metric_i, pred, gt = test_single_volume(img, img_t1, msk, model, classes=config.num_classes,
                                                    patch_size=[config.input_size_h, config.input_size_w],
                                                    test_save_path=test_save_path, case=case_name,
                                                    z_spacing=config.z_spacing, val_or_test=val_or_test)



    return 0, 0
