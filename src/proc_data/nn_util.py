import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import numpy as np

# Set seed for Python's random module
random.seed(42)

# Set seed for NumPy
np.random.seed(42)

# Set seed for PyTorch (both CPU and GPU)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Ensure deterministic behavior on GPU (optional, but slower)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_device():
    """
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class SimpleNN(nn.Module):
    def __init__(self, X_dim, hidden_dims, y_dim):
        super(SimpleNN, self).__init__()
        self.layers = nn.ModuleList()
        self.relu = nn.ReLU()
        
        prev_dim = X_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        self.output = nn.Linear(prev_dim, y_dim)
        
    def forward(self, x):
        for layer in self.layers:
            x = self.relu(layer(x))
        return self.output(x)
    

def getNNClassifier(X_dim, hidden_dims, y_dim, device, learning_rate, pos_weight=None):
    """
    """
    model = SimpleNN(X_dim=X_dim, hidden_dims=hidden_dims, y_dim=y_dim).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer


def getDataloader(X, y, batch_size, device, shuffle=False):
    """
    """
    _X = torch.tensor(X.values, dtype=torch.float32).to(device)
    _y = torch.tensor(y.values, dtype=torch.float32).to(device)
    data = TensorDataset(_X, _y.unsqueeze(1)) if len(_y.shape) == 1 else TensorDataset(_X, _y)
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)


def trainNNClassifier(model, criterion, optimizer, train_data_loader, num_epochs, print_every=0):
    """
    """
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for inputs, labels in train_data_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_data_loader.dataset)

        if print_every > 0 and ((epoch + 1) % print_every == 0 or epoch == num_epochs - 1 or epoch == 0):
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_train_loss}")
        


def fitNNClassifier(X, y, model, criterion, optimizer, device, batch_size, num_epochs, print_every=0):
    """
    """
    train_data_loader = getDataloader(X, y, batch_size, device, shuffle=True)
    trainNNClassifier(model, criterion, optimizer, train_data_loader, num_epochs, print_every=print_every)


def evaluateNNClassifier(X, y, model, criterion, device, batch_size):
    """
    """
    test_data_loader = getDataloader(X, y, batch_size, device, shuffle=False)
    test_loss = 0
    with torch.no_grad():
        model.eval()
        for inputs, labels in test_data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    avg_test_loss = test_loss / len(test_data_loader.dataset)
    print(f"Test Loss: {avg_test_loss}")
    return avg_test_loss


def predictNNClassifier(X, model, device, batch_size):
    """
    """
    _X = torch.tensor(X.values, dtype=torch.float32).to(device)
    data = TensorDataset(_X)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        model.eval()
        predictions = list()
        for inputs in data_loader:
            outputs = model(inputs[0])
            _outputs = (torch.sigmoid(outputs) > 0.5).int()
            predictions.extend(_outputs.cpu().numpy().flatten())
    
    return np.array(predictions)


if __name__ == '__main__':
    """
    time python -m src.utils.nn_utils
    """
    # from src.utils.util import X_y_split,  getRFClassifier, getAdaBoostClassifier, getLogRegClassifier, getSVMClassifier
    # from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    # from src.data.data_proc_so import load_preprocessed_data_from_disk as load_so
    # # Load data
    # _, so, so_1hot, target, train_set_indices, _, test_indices_no_duplicate = load_so()

    # # Fix the target feature 
    # so_1hot.drop(f'{target}_0', axis=1, inplace=True)
    # so_1hot.rename(columns={f'{target}_1': target}, inplace=True)

    # # Split train and test set
    # train_set, test_set = so.loc[train_set_indices], so.loc[test_indices_no_duplicate]
    # train_set_1hot, test_set_1hot = so_1hot.loc[train_set_indices], so_1hot.loc[test_indices_no_duplicate]

    # _train_set, _test_set = train_set_1hot, test_set_1hot

    # # Train the original model
    # X_train, y_train = X_y_split(df=_train_set, target=target)

    # device = get_device()
    # print(f"Using device: {device}")

    # batch_size = 10000
    # model, criterion, optimizer = getNNClassifier(X_dim=X_train.shape[1], hidden_dims=[512, 256, 128, 64, 32, 16], y_dim=1, device=device, learning_rate=0.001)
    # fitNNClassifier(X=X_train, y=y_train, model=model, criterion=criterion, optimizer=optimizer, device=device, batch_size=batch_size, num_epochs=30, print_every=10)

    # X_test, y_test = X_y_split(df=_test_set, target=target)
    # evaluateNNClassifier(X=X_test, y=y_test, model=model, criterion=criterion, device=device, batch_size=batch_size)

    # y_pred = predictNNClassifier(X=X_test, model=model, device=device, batch_size=batch_size)

    # print(y_pred.shape, y_test.shape)

    # print(y_pred[:100], np.array(y_test[:100]))

    # print(accuracy_score(y_test, y_pred))

    # print(precision_recall_fscore_support(y_test, y_pred, average='binary'))

    # # Split train and test set
    # train_set, test_set = so.loc[train_set_indices], so.loc[test_indices_no_duplicate]
    # train_set_1hot, test_set_1hot = so_1hot.loc[train_set_indices], so_1hot.loc[test_indices_no_duplicate]

    # model = 'linsvc'

    # if model == 'rf':
    #     getClassifier = getRFClassifier
    #     one_hot = False
    #     _train_set, _test_set = train_set, test_set
    # elif model == 'adaboost':
    #     getClassifier = getAdaBoostClassifier
    #     one_hot = False
    #     _train_set, _test_set = train_set, test_set
    # elif model == 'logreg':
    #     getClassifier = getLogRegClassifier
    #     one_hot = True
    #     _train_set, _test_set = train_set_1hot, test_set_1hot
    # elif model == 'linsvc':
    #     getClassifier = getSVMClassifier
    #     one_hot = True
    #     _train_set, _test_set = train_set_1hot, test_set_1hot
    
    # multi_valued = True

    # # Train the original model
    # X_train, y_train = X_y_split(df=_train_set, target=target)
    # clf = getClassifier()
    # clf.fit(X_train, y_train)

    # X_test, y_test = X_y_split(df=_test_set, target=target)
    # y_pred = clf.predict(X_test)

    # print(accuracy_score(y_test, y_pred))

    # print(precision_recall_fscore_support(y_test, y_pred, average='binary'))
    
    

