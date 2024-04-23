import torch, os, numpy as np, matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from single_pendulum import *
from plotting import *

np.set_printoptions(precision=3, suppress=True)

MAXV = 10 # maximum velocity
MAXU = 4 # maximum control input
DT = 1/60 # time step
SIMT = 10 # simulation time

def create_model(state_dim=2, input_dim=1, hidden_dim=10):
    sd, id, hd = state_dim, input_dim, hidden_dim
    model = torch.nn.Sequential(
        torch.nn.Linear(sd+id, hd),  # input layer with 2 input features and 10 hidden units
        torch.nn.Linear(hd, sd),  # output layer with 1 output unit
    )
    return model

# torch dataset
class PendulumDataset(Dataset):
    def __init__(self, dt, n):
        θs = np.random.uniform(-π, π, n)
        dθs = np.random.uniform(-MAXV, MAXV, n)
        self.x0s = np.array([θs, dθs]).T
        self.us = np.random.uniform(-MAXU, MAXU, n)
        self.dt = dt

    def __len__(self):
        return len(self.x0s)
    
    def __getitem__(self, idx):
        x0, u = self.x0s[idx], self.us[idx]
        x1 = step(x0, u, self.dt)
        input = torch.tensor(np.concatenate((x0, [u]), axis=0), dtype=torch.float32)
        output = torch.tensor(x1, dtype=torch.float32)
        return input, output
    
def nn_step(x, u, model):
    input = torch.tensor(np.concatenate((x, [u]), axis=0), dtype=torch.float32)
    output = model(input).detach().numpy()
    return output

def train_model(model):
    # hyperparameters
    n = 1500000 # number of samples
    epochs = 1 # number of epochs
    lr = 3e-2 # learning rate

    # create the model
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # create the dataset
    dataset = PendulumDataset(DT, n)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

    # train the model
    model.train()
    for epoch in range(epochs):
        for xu, nx in tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}', ncols=80, leave=False):
            optimizer.zero_grad()
            y = model(xu)
            loss = criterion(y, nx)
            loss.backward()
            optimizer.step()
        print(f'EPOCH {epoch+1}/{epochs} - LOSS: {(loss.item()):.2e}')

    model.eval()
    return model    



if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')
    
    model = create_model(state_dim=2, input_dim=1, hidden_dim=10) # create the model

    model = train_model(model)

    # initial conditions
    x0 = np.array([0.1, 0]) # initial conditions
    t = np.linspace(0, SIMT, int(SIMT/DT)) # time vector
    u = 0*np.sin(3*t) # control input
   
    # simulate the pendulum
    x = np.zeros((len(t), 2)) # state vector
    x[0] = x0 # initial conditions
    for i in range(1, len(t)): x[i] = nn_step(x[i-1], u[i], model)

    # ground truth
    xgt = np.zeros((len(t), 2)) # state vector
    xgt[0] = x0 # initial conditions
    for i in range(1, len(t)): xgt[i] = step(xgt[i-1], u[i], DT)

    # plot the results
    xs, us = np.array([x, xgt]), np.array([u, u])
    a = animate_pendulums(xs, us, DT, l, fps=60.0, figsize=(10,10), title='Pendulums')
    # a1 = animate_pendulum(x, u, DT, l, 60, (10,10), title='Neural Network')
    # a2 = animate_pendulum(xgt, u, DT, l, 60, (10,10), title='Ground Truth')
    
    plt.show()