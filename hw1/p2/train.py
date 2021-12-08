import torch
from constant import *

def train(model, device, train_loader, valid_loader, criterion, optimizer, scheduler = None, n_epochs = 100):
    best_mean_iou, best_epoch = 0, 0
    model = model.to(device)
    criterion = criterion.to(device)
    for epoch in range(n_epochs):
        print('-' * 20)
        print('Epoch {}:'.format(epoch))
        model.train()
        train_loss = []
        for batch in train_loader:
            data, target = batch
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            
        train_loss = sum(train_loss) / len(train_loss)
        print('Train Loss: {:.4f} | epoch {} / {}'.format(train_loss, epoch + 1, n_epochs))

        model.eval()
        valid_loss = []
        mean_iou = 0
        tp_fp = torch.zeros(6, device=device)
        tp_fn = torch.zeros(6, device=device)
        tp = torch.zeros(6, device=device)
        for (data, target) in valid_loader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                output = model(data)
            loss = criterion(output, target)
            valid_loss.append(loss.item())
            pred = output.max(1, keepdim=True)[1]
            pred = pred.view_as(target)
            for i in range(6):
                tp_fp[i] += torch.sum(pred == i)
                tp_fn[i] += torch.sum(target == i)
                tp[i] += torch.sum((pred == i) * (target == i))

        valid_loss = sum(valid_loss) / len(valid_loss)

        for i in range(6):
            iou = tp[i] / (tp_fp[i] + tp_fn[i] - tp[i])
            mean_iou += iou / 6
            print('class #%d : %1.5f'%(i, iou))

        print('Valid Loss: {:.4f} mIoU: {:4f} | epoch {} / {}'.format(valid_loss, mean_iou, epoch + 1, n_epochs))

        if scheduler is not None:
            scheduler.step(mean_iou)

        if (mean_iou > best_mean_iou):
            best_mean_iou, best_epoch = mean_iou, epoch
        torch.save(model.state_dict(), '{}{}_{}.ckpt'.format(chkpt_dir, model.name, epoch+1+3), _use_new_zipfile_serialization = False)

    return best_mean_iou, best_epoch
