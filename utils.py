import torch
import os
from model import FraudModel

def save_model(parameters, path="global_model.pth"):
    """
    Saves the aggregated FL parameters to a file.
    """
    print(f"Saving global model to {path}...")
    model = FraudModel()
    
    # Convert parameters to state_dict
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    
    model.load_state_dict(state_dict)
    torch.save(model.state_dict(), path)
    print("Model saved successfully.")

def load_trained_model(path="global_model.pth"):
    """
    Loads the trained model weights if they exist.
    """
    model = FraudModel()
    if os.path.exists(path):
        print(f"Loading trained weights from {path}...")
        model.load_state_dict(torch.load(path))
        model.eval()
    else:
        print("No trained weights found. Using initial random weights.")
    return model
