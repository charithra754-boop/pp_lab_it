import torch

torch.manual_seed(0)
X, y = torch.randn(100, 10), torch.randn(100, 1)

model = torch.nn.Sequential(
    torch.nn.Linear(10, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 1)
)
print(model)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    loss = loss_fn(model(X), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}  Loss = {loss.item():.4f}")

print(f"Final Loss = {loss.item():.4f}")
