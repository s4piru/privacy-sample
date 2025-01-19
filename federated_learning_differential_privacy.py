import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_non_iid_clients(X, y, num_clients, num_classes=10, classes_per_client=2):
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
        assigned_classes = classes[start_class:end_class]
        client_classes[client] = assigned_classes
    for client in range(num_clients):
        assigned_classes = client_classes[client]
        indices = []
        for cls in assigned_classes:
            indices.extend(class_indices[cls])
        np.random.shuffle(indices)
        X_client = X[indices]
        y_client = y[indices]
        client_data.append(
            (torch.from_numpy(X_client).float(), torch.from_numpy(y_client).long())
        )
    return client_data


class SimpleMLP(nn.Module):
    """A simple Multi-Layer Perceptron model with one hidden layer."""
    def __init__(self, input_dim=64, hidden_dim=32, output_dim=10):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def clip_and_add_noise_to_grads(model, clip_norm, noise_scale, device):
    """
    Clips gradient norm for each parameter and then adds noise.
    This is a simplified version of the DP-SGD approach.
    """
    # Compute total norm across all parameters
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        param_norm = param.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    # Clip gradients if needed
    clip_coef = clip_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for param in model.parameters():
            if param.grad is None:
                continue
            param.grad.data.mul_(clip_coef)

    # Add noise to gradients
    for param in model.parameters():
        if param.grad is None:
            continue
        noise = torch.randn_like(param.grad.data).to(device) * noise_scale
        param.grad.data.add_(noise)


class Client:
    def __init__(self, data, lr=0.01, device='cpu'):
        self.X, self.y = data
        self.device = device
        self.model = SimpleMLP().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)

    def set_parameters(self, global_params):
        self.model.load_state_dict(global_params)
        self.model.to(self.device)

    def local_train(
        self,
        epochs=5,
        batch_size=32,
        dp_enabled=True,
        dp_clip_norm=1.0,
        dp_noise_scale=0.0
    ):
        self.model.train()
        dataset_size = self.X.shape[0]

        for epoch in range(epochs):
            permutation = torch.randperm(dataset_size)

            for i in range(0, dataset_size, batch_size):
                batch_indices = permutation[i:i+batch_size]
                batch_X = self.X[batch_indices].to(self.device)
                batch_y = self.y[batch_indices].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()

                if dp_enabled:
                    clip_and_add_noise_to_grads(
                        self.model,
                        clip_norm=dp_clip_norm,
                        noise_scale=dp_noise_scale,
                        device=self.device
                    )

                self.optimizer.step()

    def get_parameters(self):
        """Retrieve the local model parameters."""
        return self.model.state_dict()
    

def federated_averaging(
    global_model,
    clients,
    X_test,
    y_test,
    device,
    rounds=5,
    local_epochs=5,
    fraction=1.0,
    dp_enabled=True,
    dp_clip_norm=1.0,
    dp_noise_scale=0.0
):
    global_params = global_model.state_dict()
    history = {'round': [], 'accuracy': []}

    for r in range(rounds):
        print(f"!--- Federated Round {r+1} ---!")

        num_selected = max(1, int(fraction * len(clients)))
        selected_clients = random.sample(clients, num_selected)
        print(f"Selected {num_selected} out of {len(clients)} clients.")

        local_params_list = []
        total_samples = 0

        for client in selected_clients:
            client.set_parameters(global_params)
            client.local_train(
                epochs=local_epochs,
                dp_enabled=dp_enabled,
                dp_clip_norm=dp_clip_norm,
                dp_noise_scale=dp_noise_scale
            )
            local_params_list.append((client.get_parameters(), client.X.shape[0]))
            total_samples += client.X.shape[0]

        new_global_params = {}
        for key in global_params.keys():
            weighted_sum = sum([cp[key] * num_samples for cp, num_samples in local_params_list])
            new_global_params[key] = weighted_sum / total_samples

        global_params = new_global_params
        global_model.load_state_dict(global_params)

        global_model.eval()
        with torch.no_grad():
            outputs = global_model(X_test.to(device))
            predictions = outputs.argmax(dim=1)
            accuracy = (predictions == y_test.to(device)).float().mean().item()

        history['round'].append(r+1)
        history['accuracy'].append(accuracy)

        print(f"Round {r+1} completed. Test Accuracy: {accuracy * 100:.2f}%\n")

    return global_model, history


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clients', type=int, default=10, help='Num of clients')
    parser.add_argument('--rounds', type=int, default=1000, help='Num of federated rounds')
    parser.add_argument('--local_epochs', type=int, default=5, help='Num of local epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--fraction', type=float, default=1.0, help='Ratio of clients participating each round')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--dp_clip_norm', type=float, default=1.0, help='DP clip norm for gradient clipping')
    parser.add_argument('--dp_noise_scale', type=float, default=0.0, help='Scale of noise for DP (e.g., std dev)')
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

        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            predictions = outputs.argmax(dim=1)
            accuracy = (predictions == y_test).float().mean().item()
        accuracy_history.append(accuracy)

    return accuracy_history


if __name__ == "__main__":
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

    X_test_tensor = torch.from_numpy(X_test).float().to(device)
    y_test_tensor = torch.from_numpy(y_test).long().to(device)

    clients_data = create_non_iid_clients(
        X_train, y_train, args.num_clients, num_classes=10, classes_per_client=2
    )

    clients = [Client(data, lr=args.lr, device=device) for data in clients_data]

    # Federated training with DP 
    global_model_dp = SimpleMLP().to(device)
    trained_global_model_dp, history_dp = federated_averaging(
        global_model_dp,
        clients,
        X_test_tensor,
        y_test_tensor,
        device,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        fraction=args.fraction,
        dp_enabled=True,  # Enable DP
        dp_clip_norm=args.dp_clip_norm,
        dp_noise_scale=args.dp_noise_scale
    )

    trained_global_model_dp.eval()
    with torch.no_grad():
        outputs_dp = trained_global_model_dp(X_test_tensor)
        predictions_dp = outputs_dp.argmax(dim=1)
        final_accuracy_dp = (predictions_dp == y_test_tensor).float().mean().item()


    # Federated training without DP
    global_model_nodp = SimpleMLP().to(device)
    trained_global_model_nodp, history_nodp = federated_averaging(
        global_model_nodp,
        clients,
        X_test_tensor,
        y_test_tensor,
        device,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        fraction=args.fraction,
        dp_enabled=False,
        dp_clip_norm=args.dp_clip_norm,
        dp_noise_scale=args.dp_noise_scale
    )

    trained_global_model_nodp.eval()
    with torch.no_grad():
        outputs_nodp = trained_global_model_nodp(X_test_tensor)
        predictions_nodp = outputs_nodp.argmax(dim=1)
        final_accuracy_nodp = (predictions_nodp == y_test_tensor).float().mean().item()

    smp_accuracy_history = train_single_model(
        X_train, y_train, X_test_tensor, y_test_tensor,
        device, epochs=args.rounds, lr=args.lr
    )
    final_accuracy_simple = smp_accuracy_history[-1]
    
    print(f"Final test accuracy on Digits dataset (Federated with DP): {final_accuracy_dp * 100:.2f}%")
    print(f"Final test accuracy on Digits dataset (Federated without DP): {final_accuracy_nodp * 100:.2f}%")
    print(f"Final test accuracy on Digits dataset (Single Model): {final_accuracy_simple * 100:.2f}%")

    plt.figure(figsize=(10, 6))
    plt.plot(
        history_dp['round'],
        [acc * 100 for acc in history_dp['accuracy']],
        marker='o',
        label='Federated with DP'
    )
    plt.plot(
        history_nodp['round'],
        [acc * 100 for acc in history_nodp['accuracy']],
        marker='s',
        label='Federated without DP'
    )
    plt.plot(
        range(1, len(smp_accuracy_history) + 1),
        [acc * 100 for acc in smp_accuracy_history],
        marker='x',
        label='Single Model'
    )
    plt.xlabel('Round / Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Federated Learning (with/without DP) vs. Single Model')
    plt.grid(True)
    plt.legend()
    plt.savefig('federated_dp_vs_nodp_vs_single.png')
    plt.show()
