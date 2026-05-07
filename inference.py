import torch
import pandas as pd
from model import FraudModel
from data_loader import load_data
import time

def simulate_real_time_detection():
    print("Initializing Real-Time Anomaly Detection System...")
    
    # Load model
    model = FraudModel()
    # In a real scenario, we would load the trained weights from the server
    # For demonstration, we use the initial weights
    model.eval()

    # Load a small batch of test data to simulate stream
    _, _, test_x, test_y = load_data()
    
    print("\nStarting Transaction Stream Simulation...")
    print("-" * 50)
    print(f"{'Transaction ID':<20} | {'Status':<15} | {'Probability'}")
    print("-" * 50)

    with torch.no_grad():
        for i in range(10):
            sample = test_x[i].unsqueeze(0)
            true_label = test_y[i].item()
            
            # Run inference
            output = model(sample)
            probabilities = torch.softmax(output, dim=1)
            fraud_prob = probabilities[0][1].item()
            
            status = "FRAUD" if fraud_prob > 0.5 else "NORMAL"
            
            # Print with colors (simulation)
            color_start = "\033[91m" if status == "FRAUD" else "\033[92m"
            color_end = "\033[0m"
            
            print(f"TX_{1000+i:<17} | {color_start}{status:<15}{color_end} | {fraud_prob:.4f}")
            
            time.sleep(1) # Simulate real-time delay

    print("-" * 50)
    print("Stream simulation completed.")

if __name__ == "__main__":
    simulate_real_time_detection()
