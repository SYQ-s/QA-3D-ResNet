import torch
from sklearn.metrics import accuracy_score


def train_epoch(model, criterion, optimizer, exponent_schedule, dataloader, device, epoch, logger, log_interval, writer):
    model.train()
    losses = []
    all_label = []
    all_pred = []

    for batch_idx, data in enumerate(dataloader):
        # get the inputs and labels
        inputs, labels = data['data'].to(device), data['label'].to(device)
        optimizer.zero_grad()
        # forward
        outputs = model(inputs)
        if isinstance(outputs, list):
            outputs = outputs[0]

        # compute the loss
        loss = criterion(outputs, labels.squeeze())
        losses.append(loss.item())

        # compute the accuracy
        prediction = torch.max(outputs, 1)[1]
        all_label.extend(labels.squeeze())
        all_pred.extend(prediction)
        score = accuracy_score(labels.squeeze().cpu().data.squeeze().numpy(), prediction.cpu().data.squeeze().numpy())

        # backward & optimize
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % log_interval == 0:
            logger.info("epoch {:3d} | iteration {:5d} | Loss {:.6f} | Acc {:.2f}%".format(epoch + 1, batch_idx + 1,
                                                                                           loss.item(), score * 100))
    exponent_schedule.step()

    # Compute the average loss & accuracy
    average_train_loss = sum(losses) / len(losses)
    all_label = torch.stack(all_label, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    average_train_acc = accuracy_score(all_label.squeeze().cpu().data.squeeze().numpy(),
                                       all_pred.cpu().data.squeeze().numpy())

    # add loss,acc and lr into tensorboard
    logger.info("Average Training Loss of Epoch {}: {:.6f} | Acc: {:.2f}%".format(epoch + 1, average_train_loss,
                                                                                  average_train_acc * 100))
    tags = ["accuracy", "loss", 'lr']
    writer.add_scalars(tags[0], {'average_train_acc': average_train_acc}, epoch + 1)
    writer.add_scalars(tags[1], {'average_train_loss': average_train_loss}, epoch + 1)
    writer.add_scalars(tags[2], {'learning_rate': optimizer.state_dict()['param_groups'][0]['lr']}, epoch + 1)
