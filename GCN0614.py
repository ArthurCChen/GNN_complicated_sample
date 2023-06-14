import torch
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

# Assuming X is your feature matrix and y is your adjacency matrix
X = torch.rand(320, 48)  # replace with your actual data
y = torch.randint(2, (320, 27458))  # replace with your actual data

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(48, 128)
        self.conv2 = GCNConv(128, 27458)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return torch.sigmoid(x)

import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

# Assuming X is your feature tensor and y is your adjacency matrix
X = torch.rand(320, 23, 3)  # replace with your actual data
y = torch.randint(23, (320, 27458))  # replace with your actual data

# Split the data into training and testing
train_index, test_index = train_test_split(range(320), test_size=0.25, random_state=42)

def to_edge_index(y):
    # Get all A node indices
    A_indices = torch.arange(y.size(1)).unsqueeze(0).repeat(y.size(0), 1)

    # The edge_index tensor should be of size [2, num_edges]
    edge_index = torch.stack([A_indices[y != -1], y[y != -1]], dim=0)
    return edge_index

# Create PyG Data objects for each time step
train_data = [Data(x=X[i], edge_index=to_edge_index(y[i])) for i in train_index]
test_data = [Data(x=X[i], edge_index=to_edge_index(y[i])) for i in test_index]

# Initialize the model and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training function
def train():
    model.train()
    optimizer.zero_grad()
    out = model(train_data)
    loss = F.binary_cross_entropy(out[train_data.edge_index[0]], train_data.edge_index[1])
    loss.backward()
    optimizer.step()
    return loss

# Evaluation function
def evaluate(loader):
    model.eval()

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)

    # You will want to replace this with your own evaluation metric
    return pred

# Training loop
for epoch in range(100):
    loss = train()
    print(f'Epoch: {epoch+1}, Loss: {loss:.4f}')

# Evaluation
pred = evaluate(test_data)
