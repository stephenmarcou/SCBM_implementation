import torch
from torch import nn
import os
import pickle
import time
import uuid
from pathlib import Path
import hydra
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader
import pickle

class CUBDataset(Dataset):
    def __init__(self, pkl_path):
        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f)
            self.num_concepts = len(self.data[0]["attribute_label"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Extract labels
        class_label = torch.tensor(sample["class_label"], dtype=torch.long)
        attribute_label = torch.tensor(sample["attribute_label"], dtype=torch.float)

        return attribute_label, class_label

def choose_predictor(model_type, num_concepts, num_classes):
    # Final target predictor head 
    if model_type == "linear":
        fc_y = nn.Linear(num_concepts, num_classes)
        head = nn.Sequential(fc_y)
    else:
        fc1_y = nn.Linear(num_concepts, 256)
        fc2_y = nn.Linear(256, num_classes)
        head = nn.Sequential(fc1_y, nn.ReLU(), fc2_y)

    return head


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_correct = 0
    for batch in train_loader:
        attribute_label, class_label = batch
        attribute_label, class_label = attribute_label.to(device), class_label.to(device)

        optimizer.zero_grad()
        outputs = model(attribute_label)
        predictions = torch.argmax(outputs, dim=1)
        total_correct += (predictions == class_label).sum().item()
        loss = criterion(outputs, class_label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader.dataset)
    avg_accuracy = total_correct / len(train_loader.dataset)
    return avg_loss, avg_accuracy

def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for batch in val_loader:
            attribute_label, class_label = batch
            attribute_label, class_label = attribute_label.to(device), class_label.to(device)

            outputs = model(attribute_label)
            predictions = torch.argmax(outputs, dim=1)
            total_correct += (predictions == class_label).sum().item()
            
            loss = criterion(outputs, class_label)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader.dataset)
    avg_accuracy = total_correct / len(val_loader.dataset)
    return avg_loss, avg_accuracy




def train(config):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    
    # Set paths
    timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    ex_name = "{}_{}".format(str(timestr), uuid.uuid4().hex[:5])
    pkl_file_dir = config.data.pkl_file_dir.strip("/")  
    ex_name = pkl_file_dir + "_" + ex_name

    
    experiment_path = (
        Path(config.experiment_dir) / config.model.model / config.data.dataset / ex_name
    )
    
    
    
    experiment_path.mkdir(parents=True)
    config.experiment_dir = str(experiment_path)
    print("Experiment path: ", experiment_path)
    
    
    log_file = os.path.join(experiment_path, "log.txt")
    with open(log_file, "w") as f:
        f.write(str(config) + "\n\n")  # Log the config at the beginning of the log file

    
    
    
    
    model_type = config.model.model
    pkl_dir = config.data.pkl_file_dir
    if "incomplete" in pkl_dir:
        full_data_path = os.path.join(config.data.data_path, "CUB", "incomplete_data", pkl_dir)
    else:
        full_data_path = os.path.join(config.data.data_path, "CUB", pkl_dir)
    
    
    
    train_data_path = os.path.join(full_data_path, "train.pkl")
    val_data_path = os.path.join(full_data_path, "val.pkl")
    test_data_path = os.path.join(full_data_path, "test.pkl")
    
    train_data = CUBDataset(train_data_path)
    val_data = CUBDataset(val_data_path)
    test_data = CUBDataset(test_data_path)
    
    train_loader = DataLoader(
        train_data,
        batch_size=64,
        shuffle=True,
        num_workers=4  
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=64,
        shuffle=False,
        num_workers=4 
    )
    
    test_loader = DataLoader(
        test_data,
        batch_size=64,
        shuffle=False,
        num_workers=4  
    )

    num_classes = config.data.num_classes
    log_file = os.path.join(experiment_path, "log.txt")
    info_dict = {
        "model_type": model_type,
        "num_concepts": train_data.num_concepts,
        "num_classes": num_classes,
        "pkl_dir": pkl_dir}
    
    with open(log_file, "w") as f:
        f.write(str(info_dict) + "\n\n")  # Log the config at the beginning of the log file
    
    
    
    
    pred_head = choose_predictor(model_type, train_data.num_concepts, num_classes)
    model = pred_head
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    
    for epoch in range(config.model.j_epochs):
        if epoch % config.model.validate_per_epoch == 0:
            avg_loss, avg_accuracy = validate_one_epoch(model, val_loader, criterion, device)
            print(f"Epoch {epoch+1}/{config.model.j_epochs}, Val loss: {avg_loss:.4f}, Val accuracy: {avg_accuracy:.4f}")
            with open(log_file, "a") as f:
                f.write(f"Epoch {epoch+1}/{config.model.j_epochs}, Validation Loss: {avg_loss:.4f}, Validation Accuracy: {avg_accuracy:.4f}\n")
    
        avg_loss, avg_accuracy = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{config.model.j_epochs}, Train loss: {avg_loss:.4f}, Train accuracy: {avg_accuracy:.4f}")
        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch+1}/{config.model.j_epochs}, Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_accuracy:.4f}\n")
        
    # Save model
    torch.save(model.state_dict(), os.path.join(experiment_path, "model.pth"))
    # Final evaluation on test set
    avg_loss, avg_accuracy = validate_one_epoch(model, test_loader, criterion, device)
    print(f"Final Test Loss: {avg_loss:.4f}, Final Test Accuracy: {avg_accuracy:.4f}")
    with open(log_file, "a") as f:
        f.write(f"Final Test Loss: {avg_loss:.4f}, Final Test Accuracy: {avg_accuracy:.4f}\n")
    
    
    

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    print("Configuration:")
    print(config)
    train(config)
    

    
if __name__ == "__main__":
    main()
    