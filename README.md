# Architecture-impact-of-Differential-privacy-IDS-log-moment-accounting
This experiment evaluates and compares various Private Ensemble Architectures (comprising different numbers and combinations of Teacher models transferring knowledge to a Student model) under the framework of Differential Privacy (DP), specifically analyzing the trade-offs between model utility (performance) and privacy guarantees.
Key Components of the Experiment
1. Datasets
The models are evaluated across four distinct network security and IoT-related datasets:

MQTT & Botnet (Classified as standard/binary classification tasks)
IoT & Rt_IoT (Classified as more complex, multi-class classification tasks)
2. Ensemble Architectures
The experiment tests both Homogenous (using the same type of model, e.g., all LSTMs or all CNNs) and Heterogenous (using mixed model types like LSTMs, GRUs, CNNs, and MLPs) ensembles. The structures range in scale:

Small scale: 2-3 Teachers (e.g., 3 teacher LSTM 1 student LSTM)
Medium scale: 4-6 Teachers (e.g., 4 teacher MLP,CNN,GRU,LSTM 1 student CNN)
Large scale: 10 Teachers (e.g., 10 teacher CNN 1 CNN student or mixed 10 teacher 3 LSTM, 4 MLP, 3 CNN)
3. Evaluated Metrics
To understand the trade-offs, the experiment tracks:

Utility Metrics: Accuracy (%) and F1_score (%) of the resulting student models.
Privacy Guarantees:
Data-dependent epsilon (DD-ε): The actual privacy loss calculated based on the consensus/agreement of the teachers during training (typically tighter/smaller).
Data-independent epsilon (DI-ε): The worst-case theoretical upper bound of privacy loss.
Consensus Metric: Mean vote margin representing the strength of agreement among the teacher models when voting on predictions.
Analytical Objectives Performed in the Notebook
Privacy-Utility Trade-off: Scatter plots and FacetGrids (using log scales for $\epsilon$$\epsilon$) map how accuracy and F1-score behave relative to both data-dependent and data-independent privacy budgets.
Ensemble Diversity Impact: Statistical comparison ($t$$t$-test) and visualizations (box plots/bar plots) evaluate whether Heterogeneous ensembles yield a significantly different consensus (mean vote margin) or better general performance compared to Homogeneous ones.
Scaling Effects: Analyzing how the physical number of teachers (3 vs. 6 vs. 10) influences the final utility and privacy constraints of the student.
Task Complexity Impact: Evaluating how performance differs between simpler tasks (Binary: MQTT, Botnet) versus more complex environments (Multi-class: IoT, Rt_IoT).

**Steps to recreate the experiment
**9-teacher LSTM/GRU/CNN ensemble, MLP student, seed 42, batch size 128, 23/30 epochs, learning rate \(10^{-4}\), Laplace inverse scale 1.3, \(\delta=10^{-5}\), and 32 moment orders

# =====================================================================
# 1. IMPORTS WITH DESCRIPTIVE GROUPING
# =====================================================================
import os
import sys
import copy
import math
import random
import time
import warnings
import json
import hashlib
import inspect
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warning outputs to maintain clean terminal execution logs
warnings.filterwarnings('ignore')

# =====================================================================
# 2. SEED SELECTION AND SYSTEM HYPERPARAMETERS
# =====================================================================
SEED = 42
DATA_PATH = "/content/drive/MyDrive/RT_IOT2022.csv"
TARGET_COLUMN = "Attack_type"

SELECTED_CLASSES = [
    'MQTT_Publish', 'Thing_Speak', 'Wipro_bulb', 'ARP_poisioning',
    'DDOS_Slowloris', 'DOS_SYN_Hping', 'Metasploit_Brute_Force_SSH',
    'NMAP_FIN_SCAN', 'NMAP_OS_DETECTION', 'NMAP_TCP_scan',
    'NMAP_UDP_SCAN', 'NMAP_XMAS_TREE_SCAN'
]

# Ensemble architecture definitions/do this for all configurations individually
NUM_TEACHERS = 9
TEACHER_ARCHITECTURES = ['LSTM', 'LSTM', 'LSTM', 'GRU', 'GRU', 'GRU', 'CNN', 'CNN', 'CNN']
STUDENT_ARCHITECTURE = "MLP"

# Training schedules and optimizations
BATCH_SIZE = 128
TEACHER_EPOCHS = 23
STUDENT_EPOCHS = 30
LEARNING_RATE = 1e-4
PATIENCE = 3

# Stratified split partitions mapping to 100% of data
TEACHER_VALID_FRACTION = 0.30
STUDENT_VALID_FRACTION = 0.20
TEACHER_POOL_FRACTION = 0.80
STUDENT_PRIVATE_FRACTION = 0.00
PUBLIC_FRACTION = 0.05
FINAL_TEST_FRACTION = 0.15

# Laplace Sanitization and DP moments accountant parameters
LAPLACE_INVERSE_SCALE = 1.3
DELTA = 1e-5
MAX_MOMENT = 32
USE_OVERSAMPLING = True
OUTPUT_DIR = "./pate_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Automatically route calculations to CUDA if an active GPU is present
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# 3. DATASET LOADING AND PIPELINE SPLITTING
# =====================================================================
def load_dataset(path, target_column, selected_classes):
    """
    Loads a target CSV file and filters the raw data to extract only observations
    corresponding to a specified list of network threat classes.
    """
    data = pd.read_csv(path, index_col=0)
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")
    data = data[data[target_column].isin(selected_classes)].copy()
    if len(data) == 0:
        raise ValueError("No observations remain after class filtering.")
    return data

def preprocess_and_split(data, target_column, seed=42):
    """
    Performs numeric conversions, sanitizes missing records, encodes multi-class labels,
    and executes stratified dataset partitioning to construct private, public, and test splits.
    """
    data = data.copy()
    total_fraction = TEACHER_POOL_FRACTION + STUDENT_PRIVATE_FRACTION + PUBLIC_FRACTION + FINAL_TEST_FRACTION
    if not np.isclose(total_fraction, 1.0):
        raise ValueError("Dataset fractions must sum to 1.0.")

    # Encode target labels numerically
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data[target_column])
    X = data.drop(columns=[target_column])

    # Exclude non-numerical feature descriptors
    non_numeric = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if len(non_numeric) > 0:
        X = X.drop(columns=non_numeric)

    # Enforce strict float datatypes and discard NaN records
    X = X.apply(pd.to_numeric, errors="coerce")
    valid_rows = ~X.isna().any(axis=1)
    X = X.loc[valid_rows].reset_index(drop=True)
    y = y[valid_rows.to_numpy()]

    X = X.to_numpy(dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    # Separate raw teacher pool from evaluation splits
    X_teacher, X_heldout, y_teacher, y_heldout = train_test_split(
        X, y, test_size=(1.0 - TEACHER_POOL_FRACTION), random_state=seed, stratify=y
    )

    # Sub-partition heldout records into Public/Unlabeled (Student Training) and Final Test structures
    heldout_total_sub_fraction = STUDENT_PRIVATE_FRACTION + PUBLIC_FRACTION + FINAL_TEST_FRACTION
    test_size_final_test = FINAL_TEST_FRACTION / heldout_total_sub_fraction

    X_public, X_final_test, y_public, y_final_test = train_test_split(
        X_heldout, y_heldout, test_size=test_size_final_test, random_state=seed, stratify=y_heldout
    )

    # Initialize empty arrays for student private data (set to 0.0 fraction)
    X_private = np.zeros((0, X_teacher.shape[1]), dtype=np.float32)
    y_private = np.zeros((0,), dtype=np.int64)

    return X_teacher, y_teacher, X_private, y_private, X_public, y_public, X_final_test, y_final_test, label_encoder

# =====================================================================
# 4. PYTORCH UTILITY DATASET IMPLEMENTATIONS
# =====================================================================
class TabularDataset(Dataset):
    """
    A minimal Dataset wrapper to facilitate mini-batch delivery of feature
    and label vectors through standard PyTorch DataLoaders.
    """
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, index):
        return self.features[index], self.labels[index]

def make_loader(X, y, batch_size, shuffle):
    """
    Constructs a simple, optimized DataLoader wrapper around the TabularDataset.
    """
    return DataLoader(TabularDataset(X, y), batch_size=batch_size, shuffle=shuffle, num_workers=0)

# =====================================================================
# 5. CORE ENSEMBLE TEACHER ARCHITECTURES
# =====================================================================
class TeacherLSTM(nn.Module):
    """
    Recurrent architecture using an LSTM layer for temporal patterns in sequences.
    """
    def __init__(self, input_size, hidden_size=128, num_layers=1, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        _, (hidden, _) = self.lstm(x)
        return F.log_softmax(self.fc(hidden[-1]), dim=1)

class TeacherGRU(nn.Module):
    """
    Gated Recurrent Unit architecture to process inputs sequentially.
    """
    def __init__(self, input_size, hidden_size=128, num_layers=1, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        _, hidden = self.gru(x)
        return F.log_softmax(self.fc(hidden[-1]), dim=1)

class TeacherCNN(nn.Module):
    """
    1D CNN architecture mapping localized, neighboring feature representations.
    """
    def __init__(self, input_size, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

class TeacherMLP(nn.Module):
    """
    Fully-connected Multi-Layer Perceptron architecture with regularizing dropout.
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=3):
        super().__init__()
        layers = []
        current_size = input_size
        for _ in range(num_layers):
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.20))
            current_size = hidden_size
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        if x.dim() == 3:
            x = x.mean(dim=1)
        return F.log_softmax(self.output_layer(self.network(x)), dim=1)

class StudentMLP(nn.Module):
    """
    Standard fully connected Student Multi-Layer Perceptron for PATE distillation.
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=3):
        super().__init__()
        layers = []
        current_size = input_size
        for _ in range(num_layers):
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.20))
            current_size = hidden_size
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        if x.dim() == 3:
            x = x.mean(dim=1)
        return F.log_softmax(self.output_layer(self.network(x)), dim=1)

# =====================================================================
# 6. MODEL FACTORY FUNCTIONS
# =====================================================================
def build_teacher(architecture, input_size, num_classes):
    """
    Resolves and instantiates the proper teacher model architecture based on input string.
    """
    if architecture == "LSTM":
         return TeacherLSTM(input_size=input_size, num_classes=num_classes)
    if architecture == "GRU":
         return TeacherGRU(input_size=input_size, num_classes=num_classes)
    if architecture == "CNN":
        return TeacherCNN(input_size=input_size, num_classes=num_classes)
    if architecture == "MLP":
        return TeacherMLP(input_size=input_size, num_classes=num_classes)
    raise ValueError(f"Unknown teacher architecture: {architecture}")

# =====================================================================
# 7. TRAINING, EVALUATION AND PREDICTION FUNCTIONS
# =====================================================================
def train_model(model, train_loader, valid_loader, epochs, learning_rate, patience, device):
    """
    Trains a PyTorch neural network model using cross-entropy, optimization,
    evaluation on a validation loader, and early stopping checkpoint recovery.
    """
    model = model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_validation_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history = []

    for epoch in range(epochs):
        model.train()
        total_train_loss, total_train, correct_train = 0.0, 0, 0
        for is_input, labels in train_loader:
            is_input, labels = is_input.to(device), labels.to(device)
            optimizer.zero_grad()
            log_probabilities = model(is_input)
            loss = criterion(log_probabilities, labels)
            loss.backward()
            optimizer.step()
            b_size = labels.size(0)
            total_train_loss += loss.item() * b_size
            predictions = torch.argmax(log_probabilities, dim=1)
            correct_train += (predictions == labels).sum().item()
            total_train += b_size

        model.eval()
        total_valid_loss, total_valid, correct_valid = 0.0, 0, 0
        with torch.no_grad():
            for is_input, labels in valid_loader:
                is_input, labels = is_input.to(device), labels.to(device)
                log_probabilities = model(is_input)
                loss = criterion(log_probabilities, labels)
                b_size = labels.size(0)
                total_valid_loss += loss.item() * b_size
                predictions = torch.argmax(log_probabilities, dim=1)
                correct_valid += (predictions == labels).sum().item()
                total_valid += b_size

        valid_loss = total_valid_loss / total_valid
        history.append({
            "epoch": epoch + 1,
            "train_loss": total_train_loss / total_train,
            "valid_loss": valid_loss,
            "train_accuracy": correct_train / total_train,
            "valid_accuracy": correct_valid / total_valid
        })

        # Process Early Stopping criteria
        if valid_loss < best_validation_loss:
            best_validation_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    # Restore model to its optimal parameter weights validation epoch state
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)

def predict_labels(model, loader, device):
    """
    Inferences predictions for unlabeled instances in loader using forward pass.
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            log_probabilities = model(inputs)
            batch_predictions = torch.argmax(log_probabilities, dim=1)
            predictions.extend(batch_predictions.cpu().numpy())
    return np.asarray(predictions, dtype=np.int64)

def evaluate_model(model, X_test, y_test, batch_size, device):
    """
    Evaluates accuracy, macro precision, recall, and F1 macro on test splits.
    """
    model.eval()
    loader = DataLoader(TabularDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    all_predictions = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            log_probabilities = model(inputs)
            batch_predictions = torch.argmax(log_probabilities, dim=1)
            all_predictions.extend(batch_predictions.cpu().numpy())

    all_predictions = np.array(all_predictions)
    accuracy = accuracy_score(y_test, all_predictions)
    precision = precision_score(y_test, all_predictions, average='macro', zero_division=0)
    recall = recall_score(y_test, all_predictions, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, all_predictions, average='macro', zero_division=0)
    cm = confusion_matrix(y_test, all_predictions)

    metrics = {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1_macro
    }
    return metrics, cm, all_predictions

# =====================================================================
# 8. PATE VOTE AGGREGATION UTILITIES
# =====================================================================
def teacher_predictions_to_counts(teacher_preds, num_classes):
    """
    Converts flat multi-teacher predictions arrays into structured sample-wise class bins.
    """
    teacher_preds = np.asarray(teacher_preds, dtype=np.int64)
    num_teachers, num_queries = teacher_preds.shape
    counts = np.zeros((num_queries, num_classes), dtype=np.int64)
    for query_index in range(num_queries):
        counts[query_index] = np.bincount(teacher_preds[:, query_index], minlength=num_classes)
    return counts

def aggregate_teacher_votes_laplace(teacher_preds, num_classes, laplace_inverse_scale, random_seed=None):
    """
    Adds Laplace noise to vote counts of each label to yield private consensus targets.
    """
    counts = teacher_predictions_to_counts(teacher_preds, num_classes)
    rng = np.random.default_rng(random_seed)
    laplace_scale = 1.0 / laplace_inverse_scale
    noise = rng.laplace(loc=0.0, scale=laplace_scale, size=counts.shape)
    noisy_counts = counts.astype(np.float64) + noise
    released_labels = np.argmax(noisy_counts, axis=1)
    return released_labels, counts, noisy_counts

# =====================================================================
# 9. DIFFERENTIAL PRIVACY ACCOUNTANT (RÉNYI DIFFERENTIAL PRIVACY)
# =====================================================================
def logaddexp_scalar(a, b):
    """
    Calculates log(exp(a) + exp(b)) safely avoiding numerical underflow.
    """
    if a == -math.inf: return b
    if b == -math.inf: return a
    maximum = max(a, b)
    return maximum + math.log(math.exp(a - maximum) + math.exp(b - maximum))

def logmgf_exact(q, mechanism_eps, moment_order):
    """
    Rényi Differential Privacy (RDP) log-MGF bounds derived analytically.
    """
    lam = float(moment_order)
    eps = float(mechanism_eps)
    bound_1 = 0.5 * (eps ** 2) * lam * (lam + 1.0)
    bound_3 = eps * lam
    bound_2 = math.inf
    denominator = 1.0 - math.exp(eps) * q
    if q < 0.5 and denominator > 0.0:
        log_one_minus_q = math.log1p(-q)
        log_term_1 = log_one_minus_q + lam * (log_one_minus_q - math.log(denominator))
        log_term_2 = math.log(q) + eps * lam if q > 0.0 else -math.inf
        bound_2 = logaddexp_scalar(log_term_1, log_term_2)
    return min(bound_1, bound_2, bound_3)

def compute_q_noisy_max(counts, laplace_inverse_scale):
    """
    Estimates bound on maximum probability q of an alternative label selection.
    """
    counts = np.asarray(counts, dtype=np.float64)
    num_classes = len(counts)
    if num_classes < 2:
        return 0.0
    winner = int(np.argmax(counts))
    winning_votes = counts[winner]
    competitors = np.delete(counts, winner)
    gaps = laplace_inverse_scale * (winning_votes - competitors)
    q = float(np.sum((gaps + 2.0) * np.exp(-gaps) / 4.0))
    return min(max(q, 0.0), 1.0 - 1.0 / num_classes)

def data_dependent_log_moment(counts, laplace_inverse_scale, moment_order):
    """
    Computes local data dependent log-moment for private queries.
    """
    q = compute_q_noisy_max(counts, laplace_inverse_scale)
    mechanism_eps = 2.0 * laplace_inverse_scale
    return logmgf_exact(q, mechanism_eps, moment_order)

def data_independent_log_moment(laplace_inverse_scale, moment_order):
    """
    Computes worst-case theoretical data independent log-moment.
    """
    mechanism_eps = 2.0 * laplace_inverse_scale
    lam = float(moment_order)
    concentration_bound = 0.5 * (mechanism_eps ** 2) * lam * (lam + 1.0)
    direct_dp_bound = mechanism_eps * lam
    return min(concentration_bound, direct_dp_bound)

def calculate_pate_privacy(vote_counts, laplace_inverse_scale=1.3, delta=1e-5, moments=32, expected_num_teachers=None):
    """
    Computes total data-dependent and independent epsilon bounds from query vote profiles.
    """
    vote_counts = np.asarray(vote_counts, dtype=np.int64)
    num_queries, classes_dim = vote_counts.shape
    orders = np.arange(1, moments + 1, dtype=np.float64)
    dependent_total = np.zeros(moments, dtype=np.float64)
    independent_total = np.zeros(moments, dtype=np.float64)
    q_values, vote_margins = [], []

    for counts in vote_counts:
        q = compute_q_noisy_max(counts, laplace_inverse_scale)
        q_values.append(q)
        sorted_counts = np.sort(counts)
        vote_margins.append(sorted_counts[-1] - sorted_counts[-2])
        for pos, order in enumerate(orders):
            dependent_total[pos] += data_dependent_log_moment(counts, laplace_inverse_scale, order)
            independent_total[pos] += data_independent_log_moment(laplace_inverse_scale, order)

    dep_candidates = (dependent_total - math.log(delta)) / orders
    ind_candidates = (independent_total - math.log(delta)) / orders
    dep_idx = np.argmin(dep_candidates)
    ind_idx = np.argmin(ind_candidates)

    return {
        "data_dependent_epsilon": float(dep_candidates[dep_idx]),
        "data_independent_epsilon": float(ind_candidates[ind_idx]),
        "q_values": np.asarray(q_values),
        "vote_margins": np.asarray(vote_margins)
    }

# =====================================================================
# 10. COMPLIANCE UNIT TESTS
# =====================================================================
def run_accounting_tests():
    """
    Verifies mathematical correctness of accountant outputs using established
    closed-form analytical solutions.
    """
    print("\n" + "=" * 70)
    print("RUNNING PATE ACCOUNTING COMPLIANCE TESTS")
    print("=" * 70)
    counts = np.array([10, 6, 2], dtype=float)
    eta = 0.1
    gaps = eta * np.array([4.0, 8.0])
    expected_q = min(np.sum((gaps + 2.0) * np.exp(-gaps) / 4.0), 2.0 / 3.0)
    actual_q = compute_q_noisy_max(counts, eta)
    assert np.isclose(actual_q, expected_q), f"q discrepancy: {actual_q} vs {expected_q}"
    print("PASS: Direct multiclass q verification")

    di_test_val = data_independent_log_moment(laplace_inverse_scale=1.0, moment_order=2)
    assert np.isclose(di_test_val, 4.0), f"DI log moment discrepancy: {di_test_val} vs 4.0"
    print("PASS: Independent hand-calculated DI regression test")
    print("All core compliance unit tests passed successfully!")

# =====================================================================
# 11. BASE SINGLE EXPERIMENT RUNNER
# =====================================================================
def run_full_pate_experiment(config):
    """
    Orchestrates training and privacy verification for baseline evaluation setups.
    """
    print(f"\nExecuting experiment run: {config['description']}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    data = load_dataset(DATA_PATH, TARGET_COLUMN, SELECTED_CLASSES)
    X_teacher, y_teacher, X_private, y_private, X_public, y_public, X_final_test, y_final_test, label_encoder = preprocess_and_split(data, TARGET_COLUMN, seed=SEED)

    input_size = X_teacher.shape[1]
    num_classes = len(np.unique(y_teacher))

    scaler = StandardScaler()
    X_teacher_scaled = scaler.fit_transform(X_teacher)
    X_public_scaled = scaler.transform(X_public)
    X_final_test_scaled = scaler.transform(X_final_test)

    n = len(X_teacher_scaled)
    indices = np.arange(n)
    rng = np.random.default_rng(SEED)
    rng.shuffle(indices)
    partitions = np.array_split(indices, config["num_teachers"])

    train_loaders, valid_loaders = [], []
    for teacher_id, teacher_indices in enumerate(partitions):
        X_part = X_teacher_scaled[teacher_indices]
        y_part = y_teacher[teacher_indices]
        try:
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_part, y_part, test_size=TEACHER_VALID_FRACTION, random_state=SEED + teacher_id, stratify=y_part
            )
        except ValueError:
            # Fallback to non-stratified splitting if minor classes contain too few members in this partition
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_part, y_part, test_size=TEACHER_VALID_FRACTION, random_state=SEED + teacher_id, stratify=None
            )
        if USE_OVERSAMPLING:
            sampler = RandomOverSampler(random_state=SEED + teacher_id)
            X_train, y_train = sampler.fit_resample(X_train, y_train)
        train_loaders.append(DataLoader(TabularDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True))
        valid_loaders.append(DataLoader(TabularDataset(X_valid, y_valid), batch_size=BATCH_SIZE, shuffle=False))

    teachers = []
    for idx in range(config["num_teachers"]):
        arch = config["teacher_archs"][idx % len(config["teacher_archs"])]
        model = build_teacher(arch, input_size, num_classes)
        trained_model, _ = train_model(model, train_loaders[idx], valid_loaders[idx], TEACHER_EPOCHS, LEARNING_RATE, PATIENCE, DEVICE)
        teachers.append(trained_model)

    public_loader = DataLoader(TabularDataset(X_public_scaled, y_public), batch_size=BATCH_SIZE, shuffle=False)
    all_teacher_preds = []
    for idx, t_model in enumerate(teachers):
        preds = predict_labels(t_model, public_loader, DEVICE)
        all_teacher_preds.append(preds)
    all_teacher_preds = np.stack(all_teacher_preds)

    released_labels, vote_counts, _ = aggregate_teacher_votes_laplace(all_teacher_preds, num_classes, LAPLACE_INVERSE_SCALE, random_seed=SEED)
    privacy_results = calculate_pate_privacy(vote_counts, LAPLACE_INVERSE_SCALE, DELTA, MAX_MOMENT)

    X_student_train, X_student_valid, y_student_train, y_student_valid = train_test_split(X_public_scaled, released_labels, test_size=STUDENT_VALID_FRACTION, random_state=SEED, stratify=released_labels)
    student_train_loader = DataLoader(TabularDataset(X_student_train, y_student_train), batch_size=BATCH_SIZE, shuffle=True)
    student_valid_loader = DataLoader(TabularDataset(X_student_valid, y_student_valid), batch_size=BATCH_SIZE, shuffle=False)

    student_model = StudentMLP(input_size=input_size, num_classes=num_classes)
    trained_student, _ = train_model(student_model, student_train_loader, student_valid_loader, STUDENT_EPOCHS, LEARNING_RATE, PATIENCE, DEVICE)

    final_metrics, cm, _ = evaluate_model(trained_student, X_final_test_scaled, y_final_test, BATCH_SIZE, DEVICE)

    code_str = inspect.getsource(calculate_pate_privacy) + inspect.getsource(compute_q_noisy_max)
    code_hash = hashlib.sha256(code_str.encode('utf-8')).hexdigest()

    audit_log = {
        "configuration": config["description"],
        "num_teachers": config["num_teachers"],
        "epsilon_data_dependent": privacy_results["data_dependent_epsilon"],
        "epsilon_data_independent": privacy_results["data_independent_epsilon"],
        "student_test_accuracy": final_metrics["accuracy"],
        "student_test_f1_macro": final_metrics["f1_macro"],
        "code_version_hash": code_hash
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sanitized_desc = config["description"].lower().replace(" ", "_").replace(":", "").replace("(", "").replace(")", "").replace(",", "")
    audit_record_path = os.path.join(OUTPUT_DIR, f"audit_record_{sanitized_desc}.json")
    with open(audit_record_path, "w") as f:
        json.dump(audit_log, f, indent=4)

    return {
        "teachers": teachers,
        "student": trained_student,
        "privacy_results": privacy_results,
        "final_metrics": final_metrics,
        "confusion_matrix": cm,
        "audit_record_path": audit_record_path
    }

# =====================================================================
# 12. OPTIONAL VISUALIZATION ROUTINES
# =====================================================================
def plot_privacy_loss_composition(experiment_results):
    """
    Outputs calculated Epsilon vectors onto console terminal.
    """
    priv_res = experiment_results['privacy_results']
    print(f"Data-Dependent Epsilon: {priv_res['data_dependent_epsilon']:.6f}")
    print(f"Data-Independent Epsilon: {priv_res['data_independent_epsilon']:.6f}")

# =====================================================================
# 13. ENTRYPOINT HANDLER
# =====================================================================
def main():
    print("Starting consolidated execution block...")
    run_accounting_tests()

    main_cfg = {
        'description': 'PATE: 9 Teachers (3 LSTM, 3 GRU, 3 CNN), 1 MLP Student',
        'num_teachers': NUM_TEACHERS,
        'teacher_archs': TEACHER_ARCHITECTURES,
        'student_arch': STUDENT_ARCHITECTURE
    }

    results = run_full_pate_experiment(main_cfg)
    plot_privacy_loss_composition(results)
    print("Workflow completed successfully!")

if __name__ == "__main__":
    main()



