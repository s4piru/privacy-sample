import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import defaultdict


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_non_iid_clients(X, y, num_clients, num_classes=10, classes_per_client=2, device='cpu'):
    client_data = []
    class_indices = defaultdict(list)
    
    for idx, label in enumerate(y):
        class_indices[label].append(idx)
    
    classes = np.arange(num_classes)
    np.random.shuffle(classes)
    classes_per_client = min(classes_per_client, num_classes)
    
    client_classes = {}
    for client in range(num_clients):
        start_class = client * classes_per_client
        end_class = start_class + classes_per_client
        if end_class <= num_classes:
            assigned_classes = classes[start_class:end_class]
        else:
            assigned_classes = np.concatenate((classes[start_class:], classes[:end_class % num_classes]))
        client_classes[client] = assigned_classes
    
    for client in range(num_clients):
        assigned_classes = client_classes[client]
        indices = []
        for cls in assigned_classes:
            indices.extend(class_indices[cls])
        if len(indices) == 0:
            print(f"Warning: Client {client+1} has no data assigned.")
            X_client = np.array([])
            y_client = np.array([])
        else:
            np.random.shuffle(indices)
            X_client = X[indices]
            y_client = y[indices]
        if X_client.size == 0:
            client_data.append((torch.empty(0).to(device), torch.empty(0).to(device)))
        else:
            client_data.append((torch.from_numpy(X_client).float().to(device), torch.from_numpy(y_client).long().to(device)))
    
    return client_data

def generate_additive_shares(secret, modulus=10**7, num_shares=2):
    """Splits the integer 'secret' into 'num_shares' additive shares"""
    shares = []
    sum_of_shares = 0
    for _ in range(num_shares - 1):
        r = random.randint(0, modulus - 1)
        shares.append(r)
        sum_of_shares = (sum_of_shares + r) % modulus

    last_share = (secret - sum_of_shares) % modulus
    shares.append(last_share)
    return shares

def reconstruct_secret(shares, modulus=10**7):
    return sum(shares) % modulus

def secret_share_vector(vector, modulus=10**7):
    """
    Convert a 1D float tensor to an integer representation,
    then create two additive shares for each element.
    Returns two tensors of the same shape as vector in int form.
    """
    # Flatten for iteration
    flat_vector = vector.view(-1)
    share1_list = []
    share2_list = []
    for val in flat_vector:
        int_val = int(val.item() * 1e4)
        # Map negative integers to positive range
        int_val = int_val % modulus
        s1, s2 = generate_additive_shares(int_val, modulus=modulus, num_shares=2)
        share1_list.append(s1)
        share2_list.append(s2)
    
    share1_tensor = torch.tensor(share1_list, dtype=torch.long)
    share2_tensor = torch.tensor(share2_list, dtype=torch.long)
    return share1_tensor, share2_tensor

def combine_shares_to_vector(share1_tensor, share2_tensor, modulus=10**7):
    """
    Reconstruct original float vector from two integer share tensors.
    Handles signed integers by mapping back based on modulus.
    Inverse scale from int to float.
    """
    combined = (share1_tensor + share2_tensor) % modulus
    # Map back to signed integers
    combined_signed = torch.where(
        combined >= (modulus // 2),
        combined - modulus,
        combined
    )
    float_vector = combined_signed.float() / 1e4
    return float_vector


class Aggregator:
    def __init__(self, name, modulus=10**7):
        self.name = name
        self.modulus = modulus
        self.sum_shares = None

    def reset(self):
        self.sum_shares = None

    def add_share_vector(self, share_vector):
        if self.sum_shares is None:
            self.sum_shares = share_vector.clone()
        else:
            self.sum_shares = (self.sum_shares + share_vector) % self.modulus


class SimpleMLP(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=32, output_dim=10):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class Client:
    def __init__(self, data, lr=0.01, device='cpu'):
        self.X, self.y = data
        self.device = device
        self.model = SimpleMLP().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        
    def set_parameters_from_vector(self, param_vector):
        """Load a flattened 1D float tensor into the model's state dict."""
        state_dict = self.model.state_dict()
        current_index = 0
        new_state_dict = {}
        for key, value in state_dict.items():
            num_params = value.numel()
            chunk = param_vector[current_index : current_index + num_params]
            new_state_dict[key] = chunk.view(value.shape)
            current_index += num_params
        self.model.load_state_dict(new_state_dict)

    def local_train(self, epochs=5, batch_size=32):
        """Train the local model on the client's data."""
        if self.X.numel() == 0:
            return
        self.model.train()
        dataset_size = self.X.shape[0]

        for epoch in range(epochs):
            permutation = torch.randperm(dataset_size)
            for i in range(0, dataset_size, batch_size):
                batch_indices = permutation[i : i + batch_size]
                batch_X = self.X[batch_indices].to(self.device)
                batch_y = self.y[batch_indices].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

    def get_parameters_vector(self):
        """Flatten the model parameters into a single 1D float tensor."""
        params = []
        for _, param in self.model.state_dict().items():
            params.append(param.view(-1))
        return torch.cat(params).detach().cpu()


def load_params_into_model(model, param_vector):
    """Load a single flattened parameter vector into a model's state dict."""
    state_dict = model.state_dict()
    current_index = 0
    new_state_dict = {}
    for key, value in state_dict.items():
        num_params = value.numel()
        chunk = param_vector[current_index : current_index + num_params]
        new_state_dict[key] = chunk.view(value.shape)
        current_index += num_params
    model.load_state_dict(new_state_dict)

def secure_federated_averaging(global_model, 
                               clients, 
                               aggregatorA, 
                               aggregatorB, 
                               device='cpu', 
                               rounds=3, 
                               local_epochs=1):
    """Federated Averaging where each client secret-shares its updated model parameters."""
    # Get initial global parameters as a single flattened vector
    global_params = []
    for _, param in global_model.state_dict().items():
        global_params.append(param.view(-1))
    global_vector = torch.cat(global_params).detach().cpu()

    federated_accuracies = []

    for r in range(rounds):
        print(f"!--- Federated Round {r+1} ---!")

        # Broadcast global model to clients
        for client in clients:
            client.set_parameters_from_vector(global_vector)

        # Local training
        for client in clients:
            client.local_train(epochs=local_epochs)

        # Securely aggregate local parameters
        aggregatorA.reset()
        aggregatorB.reset()

        for client_idx, client in enumerate(clients):
            # Get local param vector
            local_vector = client.get_parameters_vector()

            if local_vector.numel() == 0:
                print(f"Client {client_idx+1} has no data. Skipping secret sharing.")
                continue

            # Secret-share each client's local vector
            share1, share2 = secret_share_vector(local_vector, modulus=aggregatorA.modulus)

            # Each aggregator accumulates its share
            aggregatorA.add_share_vector(share1)
            aggregatorB.add_share_vector(share2)

            print(f"Client {client_idx+1} parameter vector securely shared.")

        # Combine aggregator shares to get the sum of local_vectors
        combined_update_vector = combine_shares_to_vector(aggregatorA.sum_shares, 
                                                          aggregatorB.sum_shares, 
                                                          modulus=aggregatorA.modulus)

        # Average the sum
        average_vector = combined_update_vector / len(clients)

        # Update the global model parameters
        global_vector = average_vector.clone()
        load_params_into_model(global_model, global_vector)  
        print(f"Round {r+1} complete.\n")

        # Optionally, evaluate global model after each round
        final_accuracy = evaluate_global_model(global_model, X_test_tensor, y_test_tensor, device=device)
        federated_accuracies.append(final_accuracy)
        print(f"Round {r+1} Test Accuracy: {final_accuracy * 100:.2f}%\n")

    # Load the final global parameters into the model
    load_params_into_model(global_model, global_vector)
    return global_model, federated_accuracies

def evaluate_global_model(model, X_test, y_test, device='cpu'):
    model.eval()
    with torch.no_grad():
        X_test_t = X_test.to(device)
        y_test_t = y_test.to(device)
        outputs = model(X_test_t)
        preds = outputs.argmax(dim=1)
        accuracy = (preds == y_test_t).float().mean().item()
    return accuracy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clients', type=int, default=10, help='Num of clients')
    parser.add_argument('--rounds', type=int, default=1000, help='Num of federated rounds')
    parser.add_argument('--local_epochs', type=int, default=5, help='Num of local epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--modulus', type=int, default=10**7, help='Modulus for additive secret sharing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()

def train_single_model(X_train, y_train, X_test, y_test, device, epochs=50, lr=0.01):
    if isinstance(X_train, np.ndarray):
        X_train = torch.from_numpy(X_train).float().to(device)
    if isinstance(y_train, np.ndarray):
        y_train = torch.from_numpy(y_train).long().to(device)

    model = SimpleMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    dataset_size = X_train.shape[0]
    accuracy_history = []

    batch_size = 32
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(dataset_size)
        
        for i in range(0, dataset_size, batch_size):
            indices = permutation[i:i+batch_size]
            batch_X = X_train[indices]
            batch_y = y_train[indices]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Evaluate on test data
        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            predictions = outputs.argmax(dim=1)
            accuracy = (predictions == y_test).float().mean().item()
        accuracy_history.append(accuracy)
        print(f"Single Model Epoch {epoch+1}/{epochs} - Test Accuracy: {accuracy * 100:.2f}%")

    return accuracy_history


if __name__ == "__main__":
    global X_test_tensor, y_test_tensor
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    digits = load_digits()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed
    )

    X_train = X_train / 16.0
    X_test = X_test / 16.0
    
    clients_data = create_non_iid_clients(
        X_train, y_train, 
        num_clients=args.num_clients, 
        num_classes=10,
        device=device
    )

    aggregatorA = Aggregator("AggregatorA", modulus=args.modulus)
    aggregatorB = Aggregator("AggregatorB", modulus=args.modulus)

    clients = [Client(data, device=device, lr=args.lr) for data in clients_data]
    global_model = SimpleMLP().to(device)

    X_test_tensor = torch.from_numpy(X_test).float().to(device)
    y_test_tensor = torch.from_numpy(y_test).long().to(device)

    trained_global_model, federated_accuracies = secure_federated_averaging(
        global_model,
        clients,
        aggregatorA,
        aggregatorB,
        device=device,
        rounds=args.rounds,
        local_epochs=args.local_epochs
    )

    final_accuracy = evaluate_global_model(trained_global_model, X_test_tensor, y_test_tensor, device=device)
    print(f"Final test accuracy on Digits dataset (Federated Learning + SMPC): {final_accuracy * 100:.2f}%")

    smp_accuracy_history = train_single_model(
        X_train, y_train, X_test_tensor, y_test_tensor,
        device, epochs=args.rounds, lr=args.lr
    )
    final_accuracy_simple = smp_accuracy_history[-1]
    print(f"Final test accuracy on Digits dataset (Single Model): {final_accuracy_simple * 100:.2f}%")

    rounds_list = list(range(1, args.rounds + 1))
    federated_accuracies_percentage = [acc * 100 for acc in federated_accuracies]
    single_accuracies_percentage = [acc * 100 for acc in smp_accuracy_history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(rounds_list, federated_accuracies_percentage, label='Federated Learning + SMPC')
    plt.plot(rounds_list, single_accuracies_percentage, label='Single Model')
    plt.xlabel('Round / Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Federated Learning + SMPC vs Single Model')
    plt.grid(True)
    plt.legend()
    plt.savefig('federated_smpc_vs_single.png')
    plt.show()
