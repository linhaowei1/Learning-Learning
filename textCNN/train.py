import model
import preprocess
from config import *
import torch
import pdb
import torch.nn as nn

def train(hyperparameter, data_loader, model, optimizer, Loss, epoch):
    model.train()
    total_loss = 0
    total_num = 0
    for batch_inputs, batch_labels in data_loader:
        optimizer.zero_grad()
        outputs, _ = model.forward(batch_inputs)
        loss = Loss(outputs, batch_labels)
        #pdb.set_trace()
        loss.backward()
        optimizer.step()
        total_loss += loss.data
        total_num += batch_labels.size(0)
    total_loss /= total_num
    print('| training | epoch {}/{} | train loss {:5.4f} '.format(epoch, epoch_num, total_loss))

def evaluate(hyperparameter, data_loader, model, Loss):
    global best_acc
    model.eval()
    total_loss = 0
    total_correct = 0
    total_num = 0
    for batch_inputs, batch_labels in data_loader:
        with torch.no_grad():
            pred_all, pred = model.forward(batch_inputs)
            total_correct += (pred == batch_labels).sum()
            total_num += pred.size(0)
            loss = Loss(pred_all, batch_labels)
            #pdb.set_trace()
            total_loss += loss.data
    acc = total_correct.item() / total_num
    total_loss /= total_num
    if best_acc < acc:
        torch.save(model.state_dict(), 'model.pkl')
        best_acc = acc
    print('| evaluation | valid loss {:5.4f} | Acc {:8.4f}\n '.format(total_loss, acc))

def main():
    #for epoch in range(1,epoch_num+1):
    #    train(hyperparameters, train_loader, model, optimizer, Loss, epoch)
    #    evaluate(hyperparameters, dev_loader, model, Loss)
    # test #
    evaluate(hyperparameters, test_loader, model, Loss)


if __name__ == '__main__':
    main()


