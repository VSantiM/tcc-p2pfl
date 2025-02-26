# pylint: disable-all
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import numpy as np


class Net(nn.Module):
    """Exemplo de rede para dados tabulares, dimensionada dinamicamente."""
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # Camada final sem ativação => logits
        return self.fc4(x)


def train_model(model, dataloader, epochs=1, learning_rate=0.01):
    """Treina o modelo utilizando CrossEntropyLoss e SGD."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    model.train()
    
    for _ in range(epochs):
        for data, target in dataloader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()


def test_model(model, dataloader):
    """Avalia o modelo (cálculo de acurácia) em um dataloader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)
            preds = torch.argmax(output, dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    return correct / total


def get_model_parameters(model):
    """
    Extrai os parâmetros do modelo como um dicionário
    (serializável em JSON), para envio a outro nó.
    """
    return {
        k: v.detach().cpu().numpy().tolist()
        for k, v in model.state_dict().items()
    }


def set_model_parameters(model, parameters):
    """
    Ajusta o state_dict do modelo a partir de um dicionário de listas/arrays.
    (ex.: recebido do servidor).
    """
    state_dict = {}
    for k, v in parameters.items():
        v_np = np.array(v, dtype=np.float32)
        expected_shape = model.state_dict()[k].shape
        if v_np.shape != tuple(expected_shape):
            raise ValueError(
                f"Shape incompatível para {k}: esperado {expected_shape}, obtido {v_np.shape}"
            )
        state_dict[k] = torch.from_numpy(v_np)

    model.load_state_dict(state_dict, strict=True)

def calculate_metrics(model, dataloader):
    """Calcula acurácia, precisão, recall e F1."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)
            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # Cálculo da matriz de confusão
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
    cm = confusion_matrix(all_targets, all_preds)
    accuracy = (np.sum(np.diag(cm)) / np.sum(cm)).round(4)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0).round(4)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0).round(4)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0).round(4)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


if __name__ == "__main__":
    # Exemplo de teste local (NÃO é o pipeline do FL).
    print("[INFO] Exemplo de uso local de task.py (apenas demonstração)")

    # Cria um dataset sintético de 100 amostras, 10 features, 4 classes
    X = torch.randn(100, 10)
    y = torch.randint(0, 4, (100,))

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Instancia a rede
    model = Net(input_dim=10, output_dim=4)

    # Treinamento rápido
    train_model(model, loader, epochs=1, learning_rate=0.01)

    # Teste (aqui mesmo dataset)
    acc = test_model(model, loader)
    print(f"[INFO] Acurácia final (exemplo sintético): {acc:.4f}")

    # Exemplo de export e import de parâmetros
    params = get_model_parameters(model)
    print("[DEBUG] Exemplo de chave do state_dict:", list(params.keys())[0])

    # Reset, recarregando parâmetros
    model2 = Net(input_dim=10, output_dim=4)
    set_model_parameters(model2, params)
    # Deve ter os mesmos parâmetros de 'model'
    print("[DEBUG] Rede model2 carregada com sucesso.")
