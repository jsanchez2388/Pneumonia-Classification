import matplotlib.pyplot as plt 
import torch.nn.functional as F 
import torch  
import numpy as np  

# Function to show an image with its label. Can also denormalize the image if needed.
def show_image(image, label, get_denormalize=True):    
    image = image.permute(1, 2, 0)  # Rearranging the image tensor dimensions from (C, H, W) to (H, W, C)
    mean = torch.FloatTensor([0.485, 0.456, 0.406])  # Mean for denormalization
    std = torch.FloatTensor([0.229, 0.224, 0.225])  # Standard deviation for denormalization

    # Denormalize the image if specified
    if get_denormalize:
        image = image * std + mean  # Apply denormalization
        image = np.clip(image, 0, 1)  # Clip values to be between 0 and 1
        plt.imshow(image)  # Display the image
        plt.title(label)  # Set the title of the plot as the label

    else:
        plt.imshow(image)  # Display the image without denormalization
        plt.title(label)  # Set the title of the plot as the label

# Function to show an image grid with an optional title.
def show_grid(image, title=None):
    image = image.permute(1, 2, 0)  # Rearranging the image tensor dimensions
    mean = torch.FloatTensor([0.485, 0.456, 0.406])  # Mean for denormalization
    std = torch.FloatTensor([0.229, 0.224, 0.225])  # Standard deviation for denormalization

    image = image * std + mean  # Apply denormalization
    image = np.clip(image, 0, 1)  # Clip values to be between 0 and 1

    plt.figure(figsize=[15, 15])  # Set the size of the figure
    plt.imshow(image)  # Display the image grid
    if title is not None:
        plt.title(title)  # Set the title if provided

# Function to calculate the accuracy of predictions.
def accuracy(y_pred, y_true):
    y_pred = F.softmax(y_pred, dim=1)  # Apply softmax to get probability distributions
    top_p, top_class = y_pred.topk(1, dim=1)  # Get the top class predictions
    equals = top_class == y_true.view(*top_class.shape)  # Compare with true labels
    return torch.mean(equals.type(torch.FloatTensor))  # Calculate the mean accuracy

# Function to view an image, its predicted probabilities, and its actual label.
def view_classify(image, ps, label):
    class_name = ['NORMAL', 'PNEUMONIA']
    classes = np.array(class_name)

    ps = ps.cpu().data.numpy().squeeze()  # Convert the predicted probabilities to numpy array

    image = image.permute(1, 2, 0)  # Rearrange image dimensions
    mean = torch.FloatTensor([0.485, 0.456, 0.406])  # Mean for denormalization
    std = torch.FloatTensor([0.229, 0.224, 0.225])  # Standard deviation for denormalization

    image = image * std + mean  # Denormalize the image
    img = np.clip(image, 0, 1)  # Clip values to be between 0 and 1

    # Creating a subplot to show image and predictions side by side
    fig, (ax1, ax2) = plt.subplots(figsize=(8, 12), ncols=2)
    ax1.imshow(img)  # Show the image
    ax1.set_title('Ground Truth : {}'.format(class_name[label]))  # Show the ground truth label
    ax1.axis('off')  # Turn off axis

    ax2.barh(classes, ps)  # Create a horizontal bar plot of predicted probabilities
    ax2.set_aspect(0.1)  # Set aspect ratio
    ax2.set_yticks(classes)  # Set y-ticks as class names
    ax2.set_yticklabels(classes)  # Set y-tick labels as class names
    ax2.set_title('Predicted Class')  # Set title
    ax2.set_xlim(0, 1.1)  # Set x-axis limits