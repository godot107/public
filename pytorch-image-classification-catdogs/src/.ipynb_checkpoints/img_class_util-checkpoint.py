import logging
import subprocess
import sys                                                                             # Python system library needed to load custom functions
import argparse
import numpy as np                                                                     # for performing calculations on numerical arrays
import pandas as pd                                                                    # home of the DataFrame construct, _the_ most important object for Data Science
import os                                                                              # for changing the directory
import math, random

import boto3

from PIL import Image


import torch
from torch import nn, optim

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Resize,  RandomHorizontalFlip, Normalize, Compose
import torchvision.models as models

class CatDogDS(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, test_set = None):
        self.animals_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.test_set = test_set
        
        if transform:
            self.transform = transform
        else:
            self.transform =  transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
        
    def __len__(self):
        return len(self.animals_df)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.root_dir,
                                self.animals_df.iloc[idx, 0])

        im = Image.open(img_path) 
        image_ts = self.transform(im)

        
        if self.animals_df.iloc[idx, 1] == 'cat':
            label = 0
        else:
            label = 1
        if self.test_set:
            return image_ts, img_path
        else:
            return image_ts, label
        

def inference (model, val_dl):
    correct_prediction = 0
    total_prediction = 0

    # Disable gradient updates
    with torch.no_grad():
        for data in val_dl:
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Get predictions
            outputs = model(inputs)

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs,1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            acc = correct_prediction/total_prediction
            # print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')
            return acc     

        
def train(model, train_dl, val_dl, num_epochs):
    
    
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    val_accuracies = []
    train_accuracies = []

    # Repeat for each epoch
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        # Repeat for each batch in the training set
        for i, data in enumerate(train_dl):
            # Get the input features and target labels, and put them on the GPU
            
            

            inputs, labels = data[0].to(device), data[1].to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs,1)

            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            # if i % 10 == 0:    # print every 10 mini-batches
                # print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))

        # Print stats at the end of the epoch
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction/total_prediction
        
        # Run Model against Validation Set
        val_acc = inference(model, val_dl)
        val_accuracies.append( (epoch, val_acc))
        train_accuracies.append( (epoch, acc))
        
        logger.info(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Training Accuracy: {acc:.2f}, Validation Accuracy: {val_acc:.2f}')
    return val_accuracies,  train_accuracies
        
        
def install(package):
    subprocess.check_call([sys.executable, "-q", "-m", "pip", "install", package])
    
    
# ----------------------------
# Inference
# ----------------------------
def inference (model, val_dl):
    correct_prediction = 0
    total_prediction = 0

    # Disable gradient updates
    with torch.no_grad():
        for data in val_dl:
          # Get the input features and target labels, and put them on the GPU
          inputs, labels = data[0].to(device), data[1].to(device)

          # Normalize the inputs
          inputs_m, inputs_s = inputs.mean(), inputs.std()
          inputs = (inputs - inputs_m) / inputs_s

          # Get predictions
          outputs = model(inputs)

          # Get the predicted class with the highest score
          _, prediction = torch.max(outputs,1)
          # Count of predictions that matched the target label
          correct_prediction += (prediction == labels).sum().item()
          total_prediction += prediction.shape[0]

        acc = correct_prediction/total_prediction
        # print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')
        return acc       

def test_predictions (model, test_dl):
     # Disable gradient updates
    file_names = []
    predictions = []
    with torch.no_grad():
        for data in test_dl:
            # Get the input features and target labels, and put them on the GPU
            
            inputs, paths = data[0].to(device), data[1]
            
            
            for path in paths:
                file_name = path.split('/')[-1]
                file_names.append(file_name)            
            

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Get predictions
            outputs = model(inputs)

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs,1)
            
            
            for i, x in enumerate(prediction):
                predictions.append(x.item())
                     
            # print(prediction)
    
    return file_names, predictions

def upload_to_s3(local_path: str, 
                 s3_path: str, 
                 bucket: str) -> str:
    """
    Upload a file from a local directory to an S3 bucket.

    Args:
        local_path (str): The path of the local file to upload.
        s3_path (str): The S3 path to upload the file to, relative to the bucket name.
        bucket (str): The name of the S3 bucket.

    Returns:
        str: The remote path to the uploaded file in the S3 bucket.

    Raises:
        None
    """
    client = boto3.client("s3")
    client.upload_file(local_path, bucket, s3_path)
    return f"s3://{bucket}/{s3_path}"

    
if __name__ == "__main__":
    
    install('torchvision') 
    from torchvision import datasets, transforms
    from torchvision.transforms import ToTensor, Resize,  RandomHorizontalFlip, Normalize, Compose
    import torchvision.models as models
    
    parser = argparse.ArgumentParser()   
    parser.add_argument("--train_batch_size", type=int, default=32)                        # training batch size
    parser.add_argument("--val_batch_size", type=int, default=64)   
    parser.add_argument("--epochs", type=int, default=10)   
    parser.add_argument("--output_data_dir", type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'),    help="Directory to save output data artifacts.")
    parser.add_argument("--data_channel", type=str, default=os.environ["SM_CHANNEL_DATA"]) # directory where input data from S3 is stored in the container
   
    parser.add_argument("--train_dir", type=str, default="train")                          # folder name with training data
    parser.add_argument("--val_dir", type=str, default="val")                              # folder name with validation data
    parser.add_argument("--test_dir", type=str, default="test")                            # folder name with test data
    parser.add_argument("--output_dir", type=str, default=os.environ['SM_MODEL_DIR'])      # output directory. This directory will be saved in the S3 bucket
    
    logger = logging.getLogger(__name__)

    args, _ = parser.parse_known_args()                    # parsing arguments from the notebook
    
    logger.info(f"Training data located at: {args.data_channel}/{args.train_dir}")
    train_path = f"{args.data_channel}/{args.train_dir}"   # directory of our training dataset on the instance
    val_path = f"{args.data_channel}/{args.val_dir}"       # directory of our validation dataset on the instance
    test_path = f"{args.data_channel}/{args.test_dir}"     # directory of our test dataset on the instance
    
    # Set up logging which allows to print information in logs


    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    logger.info("Torch version")
    logger.info(torch.__version__)
    logger.info("Torch sees CUDA?")
    logger.info(torch.cuda.is_available())
    
    ########################
    ### Prepare the Data ###
    ########################  
    
    img_transform = transforms.Compose([
        #transforms.RandomSizedCrop(224),
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    
    train_ds = CatDogDS(f"{train_path}/training_meta.csv", train_path, img_transform, None)
    val_ds = CatDogDS(f"{val_path}/val_meta.csv", val_path, None, None)
    test_ds = CatDogDS(f"{test_path}/test_meta.csv", test_path, None, True)
    
    # Create training and validation data loaders
    train_dl = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=16, shuffle=False)
    
     # Create the model and put it on the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = models.vgg16(weights=True)
    model.classifier[-1] = nn.Linear(in_features=4096, out_features= 2)
    model.to(device)
    
    # Train the model
    logger.info(f" starting training proccess for {args.epochs} epoch(s)") 
    val_accuracies, train_accuracies  = train(model, train_dl,val_dl, num_epochs = args.epochs)
    logger.info("Training Completed")
    
    # Save the model
    with open(os.path.join(args.output_dir, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)
        logger.info(f"Model saved at {args.output_dir}")
        
    # Make Test Predictions and Save    
    file_names, predictions = test_predictions(model, test_dl)

    test_predictions_df = pd.DataFrame(list(zip(file_names, predictions)),
           columns =['file_name', 'predicted_class_id'])
    
    test_predictions_df.to_csv(f"{args.output_data_dir}/prediction_test.csv", index = False)  # saving the file with test predictions
    logger.info(f"Predictions saved at {args.output_dir}")

    