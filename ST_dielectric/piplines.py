import torch
from sklearn.metrics import r2_score


def train(model, optimizer, x_train, y_train, criterion, device):
    model.train()
    x_data = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_data = torch.tensor(y_train, dtype=torch.float32, device=device)
    pred = model(x_data)
    loss = criterion(pred, y_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def validate(model, x_val, y_val, criterion, device):
    model.eval()
    with torch.no_grad():
        x_data = torch.tensor(x_val, dtype=torch.float32, device=device)
        y_data = torch.tensor(y_val, dtype=torch.float32, device=device)
        pred = model(x_data)
        loss = criterion(pred, y_data)
    return loss


def test(model, x_test, y_test, criterion, device):
    model.eval()
    with torch.no_grad():
        x_data = torch.tensor(x_test, dtype=torch.float32, device=device)
        y_data = torch.tensor(y_test, dtype=torch.float32, device=device)
        pred = model(x_data)
        loss = criterion(pred, y_data)
    return loss


def train_m(model, optimizer, x_train, y_train, criterion, device):
    model.train()
    x_data = torch.tensor(x_train[:, 1:], dtype=torch.float32, device=device)
    label = torch.tensor(x_train[:, 0].reshape(-1, 1), dtype=torch.float32, device=device)
    y_data = torch.tensor(y_train, dtype=torch.float32, device=device)
    pred = model(x_data, label)
    r2 = r2_score(y_train, pred.detach().numpy())
    loss = criterion(pred, y_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, r2


def validate_m(model, x_val, y_val, criterion, device):
    model.eval()
    with torch.no_grad():
        x_data = torch.tensor(x_val[:, 1:], dtype=torch.float32, device=device)
        label = torch.tensor(x_val[:, 0].reshape(-1, 1), dtype=torch.float32, device=device)
        y_data = torch.tensor(y_val, dtype=torch.float32, device=device)
        pred = model(x_data, label)
        loss = criterion(pred, y_data)
    return loss


def test_m(model, x_test, y_test, criterion, device):
    model.eval()
    with torch.no_grad():
        x_data = torch.tensor(x_test[:, 1:], dtype=torch.float32, device=device)
        label = torch.tensor(x_test[:, 0].reshape(-1, 1), dtype=torch.float32, device=device)
        y_data = torch.tensor(y_test, dtype=torch.float32, device=device)
        pred = model(x_data, label)
        r2 = r2_score(y_test, pred.detach().numpy())
        loss = criterion(pred, y_data)
    return loss, r2
