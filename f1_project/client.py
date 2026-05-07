import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from model import FraudModel
from data_loader import load_data
from opacus import PrivacyEngine
from captum.attr import IntegratedGradients
from sklearn.metrics import precision_score, recall_score, f1_score

class FraudClient(fl.client.NumPyClient):
    def __init__(self, client_id=0, total_clients=1):
        self.model = FraudModel()
        self.train_x, self.train_y, self.test_x, self.test_y = load_data(client_id, total_clients)
        
        # Initialize Privacy Engine for Differential Privacy
        self.privacy_engine = PrivacyEngine()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Prepare for DP (Differential Privacy)
        # We wrap the model and optimizer to add noise to gradients
        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(self.train_x, self.train_y),
                batch_size=32,
                shuffle=True
            ),
            noise_multiplier=1.1,
            max_grad_norm=1.0,
        )

    def get_parameters(self, config):
        # Handle Opacus wrapping to get clean parameters
        state_dict = self.model.state_dict()
        clean_params = []
        for k, v in state_dict.items():
            clean_params.append(v.cpu().numpy())
        return clean_params

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for (k, v), p in zip(state_dict.items(), parameters):
            state_dict[k] = torch.tensor(p)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        print("Client fitting (training with Differential Privacy)...")
        self.set_parameters(parameters)

        # Use weighted loss to handle extreme class imbalance (Fraud is ~0.17%)
        # Weighting the Fraud class (1) much higher than Normal class (0)
        weights = torch.tensor([1.0, 500.0]).to(self.train_x.device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        self.model.train()
        
        for epoch in range(5): # Increased epochs for better convergence
            epoch_loss = 0
            for batch_x, batch_y in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            print(f"Epoch {epoch+1} Loss: {epoch_loss/len(self.train_loader):.4f}")
        
        epsilon = self.privacy_engine.get_epsilon(delta=1e-5)
        print(f"Privacy Budget spent: (epsilon = {epsilon:.2f}, delta = 1e-5)")

        return self.get_parameters(config), len(self.train_x), {}

    def evaluate(self, parameters, config):
        print("Client evaluating (Anomaly Detection Metrics)...")
        self.set_parameters(parameters)
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(self.test_x)
            _, predicted = torch.max(outputs, 1)
            
            # Anomaly Detection Metrics
            acc = (predicted == self.test_y).float().mean().item()
            prec = precision_score(self.test_y.cpu(), predicted.cpu(), zero_division=0)
            rec = recall_score(self.test_y.cpu(), predicted.cpu(), zero_division=0)
            f1 = f1_score(self.test_y.cpu(), predicted.cpu(), zero_division=0)
            
            print(f"Metrics - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
            
            # Explainable AI (XAI) - Integrated Gradients
            self.explain_model()

        return float(acc), len(self.test_x), {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}

    def explain_model(self):
        print("Running XAI (Integrated Gradients) for feature importance...")
        try:
            # Create a clean copy of the model to avoid lingering Opacus hooks
            from model import FraudModel
            clean_model = FraudModel()
            
            # Extract clean state dict (remove '_module.' prefix added by Opacus)
            state_dict = self.model.state_dict()
            clean_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace('_module.', '').replace('module.', '')
                clean_state_dict[new_key] = v
            
            clean_model.load_state_dict(clean_state_dict)
            clean_model.eval()

            ig = IntegratedGradients(clean_model)
            
            # Ensure data is a tensor and requires grad
            input_data = torch.tensor(self.test_x[:5], dtype=torch.float32).requires_grad_()
            attributions = ig.attribute(input_data, target=1)
            
            # Calculate mean attribution per feature
            importance = attributions.abs().mean(dim=0)
            print(f"Top Feature Importance (abs mean): {importance[:5].tolist()}...")
        except Exception as e:
            print(f"XAI Warning: Could not complete explanation ({e})")

if __name__ == "__main__":
    # Parse CLI arguments for sharding
    cid = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    total = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    print(f"Starting Sophisticated FL Client (ID: {cid}, Total: {total})...")
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FraudClient(client_id=cid, total_clients=total)
    )
