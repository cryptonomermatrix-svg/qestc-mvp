import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sqlite3
import os

# Database for model states (persistent memory)
DB_FILE = 'ai_memory.db'
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS models (task TEXT, state BLOB)''')
conn.commit()

class SimpleNet(nn.Module):
    """Basic neural net for math operations."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Generate data for math tasks
def generate_data(task='add', size=100):
    x = np.random.rand(size, 2) * 10
    if task == 'add':
        y = x[:,0] + x[:,1]
    elif task == 'mul':
        y = x[:,0] * x[:,1]
    elif task == 'sub':  # Example extension
        y = x[:,0] - x[:,1]
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# Train with continual learning (replay buffer)
def train(model, optimizer, loss_fn, data_x, data_y, epochs=200, replay_buffers=[]):
    for epoch in range(epochs):
        model.train()
        # Replay old tasks to prevent forgetting
        for rx, ry in replay_buffers:
            out = model(rx)
            loss = loss_fn(out, ry)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Current task
        out = model(data_x)
        loss = loss_fn(out, data_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return model

# Test function
def test(model, test_x, test_y):
    model.eval()
    with torch.no_grad():
        pred = model(test_x)
        mse = nn.MSELoss()(pred, test_y)
    return mse.item()

# Save/load model state
def save_model(model, task):
    state = torch.save(model.state_dict(), 'temp.pt')
    with open('temp.pt', 'rb') as f:
        blob = f.read()
    cursor.execute("INSERT OR REPLACE INTO models (task, state) VALUES (?, ?)", (task, blob))
    conn.commit()
    os.remove('temp.pt')

def load_model(model, task):
    cursor.execute("SELECT state FROM models WHERE task=?", (task,))
    row = cursor.fetchone()
    if row:
        with open('temp.pt', 'wb') as f:
            f.write(row[0])
        model.load_state_dict(torch.load('temp.pt'))
        os.remove('temp.pt')
    return model

# Main continuous learning loop
def continuous_learning(tasks=['add', 'mul'], property_id=None):  # Tie to previous GA4 if needed
    model = SimpleNet()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()
    replay_buffers = []
    
    for i, task in enumerate(tasks):
        # Load previous state if exists
        if i > 0:
            model = load_model(model, tasks[i-1])
        
        print(f"Learning Task: {task}")
        train_x, train_y = generate_data(task, 200)
        test_x, test_y = generate_data(task, 50)
        
        # Train with replays
        model = train(model, optimizer, loss_fn, train_x, train_y, replay_buffers=replay_buffers)
        
        # Add to replay buffer (sample for efficiency)
        replay_buffers.append((train_x[:50], train_y[:50]))
        
        # Test current and previous
        current_mse = test(model, test_x, test_y)
        print(f"{task.capitalize()} MSE: {current_mse:.2f}")
        
        for prev_task in tasks[:i]:
            prev_test_x, prev_test_y = generate_data(prev_task, 50)
            prev_mse = test(model, prev_test_x, prev_test_y)
            print(f"Previous {prev_task.capitalize()} MSE after {task}: {prev_mse:.2f}")
        
        # Save state
        save_model(model, task)
    
    # Backtest: Simulate on old data
    print("\nBacktesting on historical tasks...")
    for task in tasks:
        hist_x, hist_y = generate_data(task, 50)  # Mock historical
        hist_mse = test(model, hist_x, hist_y)
        print(f"Backtest {task.capitalize()} MSE: {hist_mse:.2f}")

# Example Run (extend tasks as needed)
if __name__ == "__main__":
    continuous_learning(tasks=['add', 'mul', 'sub'])  # Add more for logic/math
