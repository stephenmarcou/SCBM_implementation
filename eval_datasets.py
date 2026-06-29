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
from torchvision import datasets
import torch.nn.functional as F
from datasets.multiclass_synthetic_dataset import load_saved_multiclass_data
from datasets.multilabel_synthetic_dataset import load_saved_multilabel_data
from utils.utils import reset_random_seeds
from datasets.cifar100_dataset_stephen import get_CIFAR100_CBM_dataloader
from datasets.CUB_dataset import get_CUB_dataloaders
from datasets.synthetic_dataset_res_scbm import get_synthetic_datasets_res_scbm, load_saved_synthetic_data




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
    
    
class CIFAR100_CBM_dataloader(datasets.CIFAR100):
    def __init__(self, *args, **kwargs):
        super(CIFAR100_CBM_dataloader, self).__init__(*args, **kwargs)
        
        # Load concepts from CIFAR-100 dataset which correspond to 20 coarse labels
        split_name = "train" if kwargs["train"] else "test"
        with open(os.path.join(kwargs["root"], "cifar-100-python", split_name), "rb") as f:
            entry = pickle.load(f, encoding="latin1")
            coarse_labels = torch.as_tensor(
                entry["coarse_labels"],
                dtype=torch.long
            )
            
            self.num_concepts = len(set(entry["coarse_labels"]))
            # Concepts shape [num_samples, 20] with one-hot encoding
            self.concepts = F.one_hot(
                coarse_labels,
                num_classes=self.num_concepts
            ).float()
    
    def __getitem__(self, idx):
        X, target = super().__getitem__(idx)
        
        return self.concepts[idx], target
        
        
        
        

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


def get_predictions(outputs, class_label, dataset, num_classes):
    """Returns (preds, probs, n_correct, n_total) for any task type."""
    
    if dataset == "multilabel_synthetic":
        probs = torch.sigmoid(outputs)          # (B, K)
        preds = (probs > 0.5).float()
        n_correct = (preds == class_label).sum().item()
        n_total = class_label.size(0) * class_label.size(1)  # n * K for Hamming
        
    elif num_classes == 2:
        probs = torch.sigmoid(outputs.squeeze(1))  # (B,)
        preds = (probs > 0.5).float()
        n_correct = (preds == class_label.float()).sum().item()
        n_total = class_label.size(0)
        
    else:
        probs = torch.softmax(outputs, dim=1)   # (B, C)
        preds = torch.argmax(outputs, dim=1)
        n_correct = (preds == class_label).sum().item()
        n_total = class_label.size(0)
    
    return preds, probs, n_correct, n_total




def train_one_epoch(config, model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0  # tracks the correct denominator for each task type
    for batch in train_loader:
        #attribute_label, class_label = batch
        class_label = batch["labels"]
        attribute_label = batch["concepts"]
        if config.model.use_residuals_from_data and "residuals" in batch:
            residuals = batch["residuals"]
            attribute_label = torch.cat([attribute_label, residuals], dim=1)
        attribute_label, class_label = attribute_label.to(device), class_label.to(device)

        optimizer.zero_grad()
        outputs = model(attribute_label)
        
        
        loss = criterion(outputs, class_label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, _, n_correct, n_total = get_predictions(
            outputs.detach(), class_label, config.data.dataset, config.data.num_classes
        )
        total_correct += n_correct
        total_samples += n_total  # N for binary/multiclass, N*K for multilabel

    avg_loss = total_loss / len(train_loader)   # sum of losses / number of batches
    avg_accuracy = total_correct / total_samples
    return {"loss": avg_loss, "accuracy": avg_accuracy}






def validate_one_epoch(config, model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_probs = []   # for AUROC, multilabel only
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            attribute_label = batch["concepts"]
            if config.model.use_residuals_from_data and "residuals" in batch:
                residuals = batch["residuals"]
                attribute_label = torch.cat([attribute_label, residuals], dim=1)
            class_label = batch["labels"]
            attribute_label, class_label = attribute_label.to(device), class_label.to(device)

            outputs = model(attribute_label)
            loss = criterion(outputs, class_label)
            total_loss += loss.item()

            _, probs, n_correct, n_total = get_predictions(
                outputs, class_label, config.data.dataset, config.data.num_classes
            )
            total_correct += n_correct
            total_samples += n_total

            if config.data.dataset == "multilabel_synthetic":
                all_probs.append(probs.cpu())
                all_labels.append(class_label.cpu())

    avg_loss = total_loss / len(val_loader)
    avg_accuracy = total_correct / total_samples
    metrics = {"loss": avg_loss, "accuracy": avg_accuracy}

    # For multilabel synthetic dataset, compute AUROC and exact match metrics
    if config.data.dataset == "multilabel_synthetic":
        from sklearn.metrics import roc_auc_score
        all_probs = torch.cat(all_probs, dim=0).numpy()    # (N, K)
        all_labels = torch.cat(all_labels, dim=0).numpy()  # (N, K)

        try:
            macro_auroc = roc_auc_score(all_labels, all_probs, average="macro")
            per_task_auroc = roc_auc_score(all_labels, all_probs, average=None).tolist()
        except ValueError:
            macro_auroc = float("nan")
            per_task_auroc = [float("nan")] * all_labels.shape[1]

        exact_match = (
            ((all_probs > 0.5) == all_labels).all(axis=1).mean()
        )
        metrics["macro_auroc"] = macro_auroc
        metrics["per_task_auroc"] = per_task_auroc
        metrics["exact_match"] = exact_match

    return metrics




def get_dataset_num_concepts(dataset):
    """Safely extract num_concepts from dataset or config.
    
    Works with both native datasets (CUB) and Subset-wrapped datasets (CIFAR) and synthetic datasets.
    """
    print(f"Number of concepts: {len(dataset[0]['concepts'])}")
    return len(dataset[0]["concepts"])





def get_dataloaders(config, gen):
    dataset = config.data.dataset
    
    if dataset == "cifar100":
        print("CIFAR-100 DATASET")
        train_data, val_data, test_data = get_CIFAR100_CBM_dataloader(
            config.data.data_path,
            gen,
            val_ratio=config.data.val_ratio,
            use_full_train_after_tuning=config.data.use_full_train_after_tuning,
        )
        

    
    elif dataset == "CUB":
        print("CUB DATASET")
    
        train_data, val_data, test_data = get_CUB_dataloaders(
            config.data, config.incomplete
        )
        
    elif dataset == "synthetic_res_scbm":
        if config.data.data_dir_name is not None:
            train_data, val_data, test_data = load_saved_synthetic_data(config)
        else:
            train_data, val_data, test_data = get_synthetic_datasets_res_scbm(config)
            
    elif dataset == "multiclass_synthetic":
        train_data, val_data, test_data = load_saved_multiclass_data(config)
        
    elif dataset == "multilabel_synthetic":
        train_data, val_data, test_data = load_saved_multilabel_data(config)
        
    return train_data, val_data, test_data





def train(config):
    # ---- Set random seeds and device ----
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    gen = reset_random_seeds(config.seed)
    
    # Prepare logging and experiment directory
    timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    ex_name = "{}_{}".format(str(timestr), uuid.uuid4().hex[:5])
    if config.data.dataset != "synthetic_res_scbm" and config.data.dataset != "multiclass_synthetic" and config.data.dataset != "multilabel_synthetic":
        pkl_file_dir = config.data.pkl_file_dir.strip("/")  
        ex_name = pkl_file_dir + "_" + ex_name

    # Create experiment directory
    if config.data.dataset == "synthetic_res_scbm":
        if config.data.data_dir_name is not None:
            ex_name = config.data.data_dir_name + f"_trueResUsed_{config.model.use_residuals_from_data}_" + ex_name
        else:
            ex_name = f"a_{config.data.alpha}_b_{config.data.beta}_g_{config.data.gamma}_trueResUsed_{config.model.use_residuals_from_data}_" + ex_name
    
        if config.data.experiment_type:
            experiment_path = (
                Path(config.experiment_dir) / config.model.model / config.data.dataset / config.data.experiment_type / ex_name
            )
        else:
            experiment_path = (
                Path(config.experiment_dir) / config.model.model / config.data.dataset / ex_name
            )

        
    
    elif config.data.dataset == "multiclass_synthetic" or config.data.dataset == "multilabel_synthetic":
        if config.data.data_dir_name is not None:
            ex_name = config.data.data_dir_name + f"_trueResUsed_{config.model.use_residuals_from_data}_" + ex_name
        
        experiment_path = (
            Path(config.experiment_dir) / config.model.model / config.data.dataset /ex_name
        )
        

    
    
    # CUB and CIFAR datasets
    else:
        experiment_path = (
            Path(config.experiment_dir) / config.model.model / config.data.dataset / ex_name
        )
        
    
    experiment_path.mkdir(parents=True)
    config.experiment_dir = str(experiment_path)
    print("Experiment path: ", experiment_path)
    
    # Create log file
    log_file = os.path.join(experiment_path, "log.txt")


    
    # ---- Load data and create dataloaders ---
    train_data, val_data, test_data = get_dataloaders(config, gen)
        
    
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
    print(train_data[0])
    num_classes = config.data.num_classes
    model_type = config.model.model
    data_type = config.data.dataset
    num_concepts = len(train_data[0]['concepts'])
    if data_type == "synthetic_res_scbm":
        num_residuals = len(train_data[0]['residuals'])
    else:
        num_residuals = 0
    

    
    
    
    

    use_residuals_from_data = config.model.use_residuals_from_data 
    
    
    if data_type == "synthetic_res_scbm" or data_type == "multiclass_synthetic" or data_type == "multilabel_synthetic":
        info_dict = {
            "model_type": model_type,
            "num_concepts": num_concepts,
            "num_residuals": num_residuals,
            "num_classes": num_classes,
            "data_dir_name": config.data.data_dir_name,
            "use_residuals_from_data": use_residuals_from_data
        }
    
    else:  
        info_dict = {
            "model_type": model_type,
            "num_concepts": num_concepts,
            "num_residuals": num_residuals,
            "num_classes": num_classes,
            "pkl_file_dir": pkl_file_dir,
            "use_residuals_from_data": use_residuals_from_data
            }
        
    with open(log_file, "w") as f:
        f.write(str(info_dict) + "\n\n")  # Log the config at the beginning of the log file
    
    

    if data_type == "synthetic_res_scbm" or data_type == "multiclass_synthetic" or data_type == "multilabel_synthetic":
        open_data_log_file_and_write_info(config, log_file, config.data.data_dir_name)
    
    
    # ---- Prepare model ----
    if config.model.use_residuals_from_data:
        num_concepts += num_residuals
    pred_head = choose_predictor(model_type, num_concepts, num_classes)
    model = pred_head
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = choose_loss(config, num_classes)
    
    
    for epoch in range(config.model.j_epochs):
        if epoch % config.model.validate_per_epoch == 0:
            metrics = validate_one_epoch(config, model, val_loader, criterion, device)
            log_metrics(config, epoch, metrics, log_file, validation=True)
            

                
        metrics = train_one_epoch(config, model, train_loader, optimizer, criterion, device)
        log_metrics(config, epoch, metrics, log_file, validation=False)
        
    # Save model
    torch.save(model.state_dict(), os.path.join(experiment_path, "model.pth"))

    # Final evaluation on test set
    metrics = validate_one_epoch(config, model, test_loader, criterion, device)

    if config.data.dataset == "multilabel_synthetic":
        macro_auroc = metrics.get("macro_auroc", float("nan"))
        exact_match = metrics.get("exact_match", float("nan"))
        per_task_auroc = metrics.get("per_task_auroc", [])
        task_str = ", ".join(f"Task{i}: {auc:.4f}" for i, auc in enumerate(per_task_auroc))
        msg = (
            f"Final Test Loss: {metrics['loss']:.4f}, "
            f"Hamming Acc: {metrics['accuracy']:.4f}, "
            f"Macro AUROC: {macro_auroc:.4f}, "
            f"Exact Match: {exact_match:.4f}\n"
            f"  Per-task AUROC — {task_str}"
        )
    else:
        msg = (
            f"Final Test Loss: {metrics['loss']:.4f}, "
            f"Final Test Accuracy: {metrics['accuracy']:.4f}"
        )

    print(msg)
    with open(log_file, "a") as f:
        f.write(msg + "\n")

    
    
    
def log_metrics(config, epoch, metrics, log_file, validation=False):
    dataset_type = "Val" if validation else "Train"

    avg_loss = metrics["loss"]
    avg_accuracy = metrics["accuracy"]

    if config.data.dataset != "multilabel_synthetic":
        msg = (
            f"Epoch {epoch+1}/{config.model.j_epochs}, "
            f"{dataset_type} Loss: {avg_loss:.4f}, "
            f"{dataset_type} Accuracy: {avg_accuracy:.4f}"
        )
    else:
        msg = (
            f"Epoch {epoch+1}/{config.model.j_epochs}, "
            f"{dataset_type} Loss: {avg_loss:.4f}, "
            f"{dataset_type} Hamming Acc: {avg_accuracy:.4f}"
        )
        if validation:
            macro_auroc = metrics.get("macro_auroc", float("nan"))
            exact_match = metrics.get("exact_match", float("nan"))
            per_task_auroc = metrics.get("per_task_auroc", [])
            msg += f", Macro AUROC: {macro_auroc:.4f}, Exact Match: {exact_match:.4f}"
            if per_task_auroc:
                task_str = ", ".join(
                    f"Task{i}: {auc:.4f}" for i, auc in enumerate(per_task_auroc)
                )
                msg += f"\n  Per-task AUROC — {task_str}"

    print(msg)
    with open(log_file, "a") as f:
        f.write(msg + "\n")
    
    
    
    
    
def open_data_log_file_and_write_info(config, log_file, data_dir_name):
    if config.data.dataset != "synthetic_res_scbm" and config.data.dataset != "multiclass_synthetic" and config.data.dataset != "multilabel_synthetic":
        raise ValueError("Only synthetic datasets are supported for writing this info")
    
    if config.data.dataset == "synthetic_res_scbm":
        if config.data.experiment_type:
            data_dir_full_path = os.path.join(config.data.data_path, "synthetic_res_scbm", config.data.experiment_type, data_dir_name)
        else:
            data_dir_full_path = os.path.join(config.data.data_path, "synthetic_res_scbm", data_dir_name)
            
    elif config.data.dataset == "multiclass_synthetic":
        data_dir_full_path = os.path.join(config.data.data_path, "multiclass_synthetic", data_dir_name)
        
        
    elif config.data.dataset == "multilabel_synthetic":
        data_dir_full_path = os.path.join(config.data.data_path, "multilabel_synthetic", data_dir_name)
        
    
        
    info_file = os.path.join(data_dir_full_path, "info.txt")
    with open(log_file, "a") as f:
        with open(info_file, "r") as info_f:
            info_content = info_f.read()
            f.write(f"Synthetic dataset info:\n{info_content}\n\n")
        
        

def choose_loss(config, num_classes):
    if num_classes == 2 or config.data.dataset == "multilabel_synthetic":
        return nn.BCEWithLogitsLoss()
    else:
        return nn.CrossEntropyLoss()
    

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig) -> None:

    print("Configuration:")
    print(config)
    train(config)
    
    

    
if __name__ == "__main__":
    main()
    