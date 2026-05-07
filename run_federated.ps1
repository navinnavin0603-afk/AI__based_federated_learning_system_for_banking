# run_federated.ps1
# Start the server in a new window
Write-Host "Starting Server..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "python server.py"

Write-Host "Waiting 8 seconds for server to initialize..."
Start-Sleep -Seconds 8

# Start Client 1
Write-Host "Launching Client 1 (Shard 0/3)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "python client.py 0 3"

# Start Client 2
Write-Host "Launching Client 2 (Shard 1/3)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "python client.py 1 3"

# Start Client 3
Write-Host "Launching Client 3 (Shard 2/3)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "python client.py 2 3"

Write-Host "Federated system is now running with 1 Server and 3 Sharded Clients." -ForegroundColor Magenta
