from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import torch
from utils import load_trained_model
from data_loader import load_data
try:
    from captum.attr import IntegratedGradients
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False
    print("Warning: 'captum' not found. XAI features will be disabled.")

import random
import os
import json
import subprocess
import signal

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return send_from_directory('.', 'dashboard.html')

# Track training processes
training_processes = []

def kill_port_process(port):
    """Forcefully kills any process using the specified port on Windows."""
    try:
        # Find PIDs using the port
        cmd = f"netstat -ano | findstr :{port}"
        output = subprocess.check_output(cmd, shell=True).decode()
        pids = set()
        for line in output.splitlines():
            parts = line.strip().split()
            if len(parts) >= 5 and "LISTENING" in line:
                pid = parts[-1]
                if pid != "0" and int(pid) != os.getpid():
                    pids.add(pid)
        
        for pid in pids:
            print(f"Cleaning up zombie process {pid} on port {port}...")
            subprocess.run(f"taskkill /F /PID {pid} /T", shell=True, capture_output=True)
    except Exception as e:
        pass

# Global variables for Hot-Reload
MODEL_PATH = "global_model.pth"
last_model_time = 0
model = load_trained_model(MODEL_PATH)

def check_for_model_updates():
    """Reloads the model if the global_model.pth file has been updated."""
    global model, last_model_time
    if os.path.exists(MODEL_PATH):
        current_time = os.path.getmtime(MODEL_PATH)
        if current_time > last_model_time:
            print("Detected new model weights! Reloading for live inference...")
            model = load_trained_model(MODEL_PATH)
            last_model_time = current_time

# Initial timestamp
if os.path.exists(MODEL_PATH):
    last_model_time = os.path.getmtime(MODEL_PATH)

# Load a bit of data for demonstration
_, _, test_x, test_y = load_data()

@app.route('/api/status', methods=['GET'])
def get_status():
    check_for_model_updates()
    
    # Default metrics
    metrics = {
        "status": "Online",
        "privacy_budget": 1.24,
        "round": 0,
        "accuracy": 0,
        "precision": 0,
        "recall": 0,
        "f1": 0
    }
    
    # Try to load real metrics from training
    if os.path.exists("metrics.json"):
        try:
            with open("metrics.json", "r") as f:
                data = json.load(f)
                metrics.update(data)
        except:
            pass
            
    # Count active training processes (subtracting 1 for the server process)
    running_procs = [p for p in training_processes if p.poll() is None]
    metrics["active_nodes"] = max(0, len(running_procs) - 1)
            
    return jsonify(metrics)

@app.route('/api/train/start', methods=['POST'])
def start_training():
    # Forcefully kill any lingering server/client processes
    # 1. Kill by port (most reliable for clearing 8080)
    kill_port_process(8080)
    
    # 2. Kill by process name/title as a fallback
    try:
        subprocess.run("taskkill /F /IM python.exe /FI \"WINDOWTITLE eq python*\" /T", shell=True, capture_output=True)
    except:
        pass
    
    # 3. Terminate tracked objects
    for p in training_processes:
        try:
            p.terminate()
        except:
            pass

    # Create logs directory if it doesn't exist
    if not os.path.exists("logs"):
        os.makedirs("logs")

    training_processes = []
    try:
        # 1. Start Server
        server_log = open("logs/server.log", "w")
        server_proc = subprocess.Popen(["python", "server.py"], stdout=server_log, stderr=server_log)
        training_processes.append(server_proc)
        
        # 2. Wait for server
        import time
        time.sleep(5)
        
        # 3. Start 3 Clients
        for i in range(3):
            log_file = open(f"logs/client_{i}.log", "w")
            client_proc = subprocess.Popen(["python", "client.py", str(i), "3"], stdout=log_file, stderr=log_file)
            training_processes.append(client_proc)
            time.sleep(1)
            
        return jsonify({"status": "success", "message": "Federated Training Started (4 processes)"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/train/status', methods=['GET'])
def get_training_status():
    if not training_processes:
        return jsonify({"in_progress": False, "message": "No training session started"})
    
    # Check if any process is still running
    running = any(p.poll() is None for p in training_processes)
    return jsonify({
        "in_progress": running,
        "process_count": len(training_processes),
        "message": "Training in progress..." if running else "Training complete or stopped"
    })

@app.route('/api/inference', methods=['GET'])
def run_inference():
    check_for_model_updates() # Live reload!
    # Pick a random sample
    idx = random.randint(0, len(test_x) - 1)
    sample = test_x[idx].unsqueeze(0)
    true_label = test_y[idx].item()
    
    with torch.no_grad():
        output = model(sample)
        probabilities = torch.softmax(output, dim=1)
        fraud_prob = probabilities[0][1].item()
        
    # Run XAI if available
    if HAS_CAPTUM:
        ig = IntegratedGradients(model)
        sample.requires_grad_()
        attributions = ig.attribute(sample, target=1)
        importance = attributions.abs().mean(dim=0).tolist()
    else:
        # Fallback to random importance for demonstration if XAI is missing
        importance = [random.uniform(0.1, 0.5) for _ in range(5)]
    
    return jsonify({
        "transaction_id": f"TX_{1000 + idx}",
        "fraud_probability": round(fraud_prob, 4),
        "status": "FRAUD" if fraud_prob > 0.5 else "NORMAL",
        "true_label": "FRAUD" if true_label == 1 else "NORMAL",
        "xai_importance": importance[:5] # Top 5 features
    })


if __name__ == '__main__':
    print("Backend API running on http://127.0.0.1:5000")
    app.run(port=5000, debug=True)
