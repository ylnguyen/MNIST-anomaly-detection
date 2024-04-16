import torch
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

from loss_functions import *

def add_noise(image, noise_factor=0.6):
    """
    Adds noise to a tensor and normalizes the image.
    Params:     image (tensor)
                noise_factor (float): states the amount of noise added
    """
    noisy_image = image + noise_factor * torch.randn(image.shape)
    return np.clip(noisy_image, 0., 1.)

def train(encoder, decoder, dataloader, loss_fn, optimizer):
    """
    Training loop for the autoencoders
    Params:     encoder (torch.nn.Module): encoder model
                decoder (torch.nn.Module): decoder model
                dataloader (DataLoader object): training dataloader
                loss_fn (function): the loss function
                optimizer (torch.optim): optimizer
    """
    # Set models to train mode
    encoder.train()
    decoder.train()
    # save loss to list
    train_loss = []
    
    for image_batch, _ in dataloader:           
        # Encode data
        encoded_data = encoder(image_batch)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass to update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # save loss to list
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss), train_loss

def validation(encoder, decoder, dataloader, loss_fn):
    """
    Validation loop for the autoencoders
    Params:     encoder (torch.nn.Module): encoder model
                decoder (torch.nn.Module): decoder model
                dataloader (DataLoader object): training dataloader
                loss_fn (function): the loss function
    """
    # Set to evaluation mode
    encoder.eval()
    decoder.eval()
    
    # Turn tracking gradients off
    with torch.no_grad():
        # Define the lists to store the outputs for each batch
        outputs = []
        labels = []
        
        for image_batch, _ in dataloader:
            # Encode data
            encoded_data = encoder(image_batch)
            # Decode data
            decoded_data = decoder(encoded_data)
            # Append the reconstruction and the original image to the lists
            outputs.append(decoded_data)
            labels.append(image_batch)
        # Create a single tensor with all the values in the lists
        outputs = torch.cat(outputs)
        labels = torch.cat(labels) 
        # Evaluate loss over all batches
        val_loss = loss_fn(outputs, labels)
    return val_loss.data

def test(encoder, decoder, dataloader, loss_fn):
    """
    Test loop for the autoencoders
    Params:     encoder (torch.nn.Module): encoder model
                decoder (torch.nn.Module): decoder model
                dataloader (DataLoader object): training dataloader
                loss_fn (function): the loss function
    """
    # Define the lists to store the outputs for each batch
    outputs = []
    noisy_outputs = []
    inputs = []
    noisy_inputs = []

    test_loss = []
    test_loss_noise = []
    
    encoder.eval()
    decoder.eval()
    for image_batch, _ in dataloader:
        # apply noise to batch
        noisy_batch = add_noise(image_batch.clone()) 
        # feed normal images through autoencoder
        encoded_data = encoder(image_batch.clone())
        decoded_data = decoder(encoded_data)
        # feed noisy counterpart batch
        encoded_noise = encoder(noisy_batch)
        decoded_noise = decoder(encoded_noise)
    
        # calculate loss
        n_loss = loss_fn(decoded_noise, noisy_batch)
        test_loss_noise.append(float(n_loss.detach().cpu().numpy()))
    
        # save input images
        inputs.append(image_batch)
        outputs.append(decoded_data)
        # save reconstructions
        noisy_inputs.append(noisy_batch)   
        noisy_outputs.append(decoded_noise)
    
    return [noisy_outputs, noisy_inputs, outputs, inputs], [test_loss, test_loss_noise]

def plot_losses(losses, loss_fn='MSE'):
    """
    Plot the training and validation batch loss per epoch
    """
    plt.figure(figsize=(10,8))
    # plot the curves
    plt.semilogy(losses['avg_train_loss'], label='train')
    plt.semilogy(losses['val_loss'], label='validation')
    # Set X and Y labels
    plt.xlabel('Epoch [-]', fontsize=20)
    plt.ylabel('Average Loss[-]', fontsize=20)
    # Increase font size
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=25)
    
    plt.legend(fontsize="x-large")
    plt.title(f'{loss_fn} loss', fontsize=25)
    plt.show()
    
def show_noisy_images(dataloader, num_imgs=4):
    """
    Visualize input images and the corrupted counterparts on the test set.
    Used for question 1.
    """
    data_iter = iter(dataloader)
    images, _ = next(data_iter)
    
    fig = plt.figure()
    for i in range(num_imgs):
        # add subplot
        ax1 = fig.add_subplot(2, num_imgs, i+1)
        ax2 = fig.add_subplot(2, num_imgs, i+5)
        # add noise (exercise 1.1)
        im_noise = add_noise(images[i])
        # get dimensions
        n,m = im_noise.shape[1], im_noise.shape[2]
        # visualize normal images
        ax1.imshow(images[i].reshape(n, m), cmap='gray')
        ax1.axis('off')
        # visualize noisy images
        ax2.imshow(im_noise.reshape(n, m), cmap='gray')
        ax2.axis('off')
    fig.tight_layout()
    # Add titles
    plt.figtext(0.5,0.95, "Original", ha="center", va="top", fontsize=14)
    plt.figtext(0.5,0.5, "Corrupted", ha="center", va="top", fontsize=14)
    fig.suptitle('Comparison of test samples', fontsize=16, y=1.05)
    
def test_images(yhat_corrupt, x_corrupt, yhat, x, num_imgs=4):
    fig = plt.figure(figsize=(20,10))
    for i in range(num_imgs):
        # add subplot
        ax1 = fig.add_subplot(num_imgs, 10, i+1)
        ax2 = fig.add_subplot(num_imgs, 10, i+11)
        ax3 = fig.add_subplot(num_imgs, 10, i+21)
        ax4 = fig.add_subplot(num_imgs, 10, i+31)
        # get dimensions
        n,m = x[0][i].shape[1], x[0][i].shape[2]
        
        # visualize noisy images
        im_corrupt = x_corrupt[0][i].detach().numpy()
        ax1.imshow(im_corrupt.reshape(n, m), cmap='gray')
        ax1.axis('off')
        # visualize reconstruction of noisy images
        pred_corrupt = yhat_corrupt[0][i].detach().numpy()
        ax2.imshow(pred_corrupt.reshape(n, m), cmap='gray')
        ax2.axis('off')
        # visualize normal images
        im_true = x[0][i].detach().numpy()
        ax3.imshow(im_true.reshape(n, m), cmap='gray')
        ax3.axis('off')
        # visualize reconstruction of normal images
        pred = yhat[0][i].detach().numpy()
        ax4.imshow(pred.reshape(n, m), cmap='gray')
        ax4.axis('off')
            
    fig.tight_layout()
    # Add titles
    plt.figtext(0.5,0.9, "Corrupted", ha="center", va="top", fontsize=35)
    plt.figtext(0.5,0.65, "Reconstruction", ha="center", va="top", fontsize=35)
    plt.figtext(0.5,0.40, "Original", ha="center", va="top", fontsize=35)
    plt.figtext(0.5,0.15, "Reconstruction", ha="center", va="top", fontsize=35)
    fig.suptitle('Comparison of test samples', fontsize=45, x = 0.3, y=1.05)
    
def calculate_performance(yhat, y):
    """
    Compare the prediction with label to get the TP, FP, TN and FN values.
    In addition, the precision, recall and F1 score will be calculated.
    Parameters:     yhat (list): contains the predictions 0 or 1
                    y (list): contains the true label 0 or 1
    """
    # Initialize counters for confusion matrix components
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    # We go through all images and compare the prediction and ground truth label
    for i in range(len(yhat)): 
        # True positives
        if y[i]==yhat[i]==1:
            TP += 1
        # False positives
        if yhat[i]==1 and y[i]!=yhat[i]:
            FP += 1
        # True negatives
        if y[i]==yhat[i]==0:
            TN += 1
        # False negatives
        if yhat[i]==0 and y[i]!=yhat[i]:
            FN += 1
    
    print(f'TP:{TP}\nFP:{FP}\nTN:{TN}\nFN:{FN}')
    
    # Calculate precision, recall and F1 score
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    if (precision!=0) or (recall!=0):
        f1 = 2*(precision*recall)/(precision + recall)
    # In case precision or recall is zero, set F1 to nan
    else: f1 = float("nan")
    
    print(f'Precision: {precision:.2f}\nRecall: {recall:.2f}\nF1: {f1:.2f}')
    return [TP, FN, FP, TN], [precision, recall, f1]

def get_roc_auc(preds, labels):
    """
    Plots the ROC curve for the prediction
    Parameters:     yhat (list): contains the predictions 0 or 1
                    y (list): contains the true label 0 or 1
    """
    # Use scikit-learn to get the false positive and true positive rate
    fpr, tpr, thresholds = roc_curve(labels, preds)
    # Calculate the Area under the Curve
    auc = 1 * np.trapz(tpr, fpr)
    
    plt.figure()
    # Plot FPR versus TPR
    plt.plot(fpr, tpr, linestyle='--', marker='o', color='darkorange', lw = 2, label='ROC curve', clip_on=False)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    # Add X and Y labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # Add title
    plt.title(f'ROC curve, AUC = {auc:.2f}')
    plt.legend(loc="lower right")
    plt.show()