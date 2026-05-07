# run_full_stack.ps1

# 1. Install dependencies
Write-Host "--- Step 1: Installing dependencies ---" -ForegroundColor Cyan
pip install -r requirements.txt

# 2. Start the Flower Server
Write-Host "--- Step 2: Starting Flower Server ---" -ForegroundColor Cyan
Start-Process powershell -ArgumentList "python server.py" -NoNewWindow
Start-Sleep -Seconds 5

# 3. Start the Flower Client (Training)
Write-Host "--- Step 3: Starting Flower Client (Training) ---" -ForegroundColor Cyan
Start-Process powershell -ArgumentList "python client.py" -NoNewWindow
Start-Sleep -Seconds 2

# 4. Start the Backend API (Flask)
Write-Host "--- Step 4: Starting Backend API (Connection Bridge) ---" -ForegroundColor Cyan
Start-Process powershell -ArgumentList "python app.py" -NoNewWindow
Start-Sleep -Seconds 3

# 5. Launch the Dashboard
Write-Host "--- Step 5: Launching Dashboard ---" -ForegroundColor Cyan
Start-Process "dashboard.html"

Write-Host "`nAll systems online! Check your browser for the dashboard." -ForegroundColor Green
Write-Host "To stop everything, close the PowerShell windows or press Ctrl+C here (Note: background processes might need manual closing)."
