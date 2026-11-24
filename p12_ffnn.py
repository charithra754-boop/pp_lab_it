import torch
import torch.nn as nn

torch.manual_seed(0)

X = torch.randn(100, 10)
y = torch.randn(100, 1)

model = nn.Sequential(
    nn.Linear(10, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    pred = model(X)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}  Loss = {loss.item():.4f}")

print(f"Final Loss = {loss.item():.4f}")

# python3 p12_ffnn.py
