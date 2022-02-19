import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

def calculateMetrics(y_true, y_pred, epoch_no):
    #y_true, y_pred = y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
    y_pred = np.array(y_pred) >= .5
    CM = confusion_matrix(y_true, y_pred)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    snstv = TP / (TP + FN)
    spcft = TN / (TN + FP)
    res = []
    res.append([snstv, spcft, accuracy_score(y_true, y_pred)])
    result_df = pd.DataFrame(res,columns = ['sensitivity','specificity', 'accuracy'])
    print(result_df)
    result_df.to_csv('results/result_' + str(epoch_no) + '.csv')


def train_function(model, train_loader, valid_loader, criterion, optimizer, epoch, device=None, scheduler=None, train_on_gpu=True):
    #valid_loss_min = 0.218098#np.Inf
    valid_loss_min = 1.0
    if train_on_gpu:
        model.to(device)
    train_loss = 0.0
    valid_loss = 0.0
    if scheduler != None:
        scheduler.step()
    model.train()
    for data, target in tqdm(train_loader):
        if train_on_gpu:
            data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = torch.squeeze(model(data))
        loss = criterion(output, target.float())
        loss.backward()
        optimizer.step()   
        train_loss += loss.item() * data.size(0)
    ######################    
    # validate the model #
    ######################
    model.eval()
    number_correct, number_data = 0, 0
    all_preds = []
    all_gts = []
    for data, target in tqdm(valid_loader):
        if train_on_gpu:
            data, target = data.to(device), target.to(device)
        output = torch.squeeze(model(data))
        loss = criterion(output, target.float())
        valid_loss += loss.item() * data.size(0)
        ############# calculate the accurecy
        #_, pred = torch.max(output, 1) 
        pred = output
        correct_tensor = pred.eq(target.data.view_as(pred))   
        all_preds = all_preds + pred.detach().cpu().numpy().tolist()
        all_gts = all_gts + target.data.tolist()
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu \
                                else np.squeeze(correct_tensor.cpu().numpy())
        number_correct += sum(correct)
        number_data += correct.shape[0]
    calculateMetrics(all_gts, all_preds, epoch)
    ###################################
    train_loss = train_loss / len(train_loader.dataset)
    valid_loss = valid_loss / len(valid_loader.dataset)
    accuracy = (100 * number_correct / number_data)
    print('Epoch: {} \n-----------------\n \tTraining Loss: {:.6f} \t Validation Loss: {:.6f} \t accuracy : {:.4f}% '.format(epoch, train_loss, valid_loss,accuracy))
    model.to(device)
    return valid_loss
    
def save_checkpoint(epoch, epoch_since_improvement, model, optimizer, loss, best_loss, is_best, epoch_no):
    state = {
                'epoch' : epoch,
                'epoch_since_improvement' : epoch_since_improvement,
                'loss' : loss,
                'best_loss' : best_loss,
                'model' : model,
                'optimizer' : optimizer
            }
    path = '/home/dcl/Desktop/TrustableAIProject/'
    filename = 'checkpoint_' + str(epoch_no) + '.pth.tar'
    torch.save(state, path + filename)
    if is_best:
        torch.save(state, path + 'BEST_checkpoint.pth.tar')
        
