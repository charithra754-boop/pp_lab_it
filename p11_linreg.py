import torch

torch.manual_seed(0)
x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = 2 * x + 1

model = torch.nn.Linear(1, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    loss = loss_fn(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d}  Loss = {loss.item():.4f}  dL/dw = {model.weight.grad.item():.4f}  dL/db = {model.bias.grad.item():.4f}")

print(f"\nLearned: w = {model.weight.item():.3f}, b = {model.bias.item():.3f}  (true: w = 2, b = 1)")
