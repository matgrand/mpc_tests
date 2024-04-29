import torch, os, numpy as np, matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import lightning as L

from single_pendulum import *
from plotting import *

MAXV = 8 # maximum velocity
MAXU = 4 # maximum control input
DT = 1/60 # time step
N_SAMPLES = 10000

SIMT = 15 # simulation time
    
# torch dataset
class Pendulum1StepDataset(Dataset):
    def __init__(self, dt, n):
        self.dt = dt
        self.n = n

    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        θ0, dθ0, u = np.random.uniform(-π, π), np.random.uniform(-MAXV, MAXV), np.random.uniform(-MAXU, MAXU)
        x1 = step(np.array([θ0, dθ0]), u, self.dt, wa=True)
        input = torch.tensor([θ0, dθ0, u], dtype=torch.float32)
        output = torch.tensor(x1, dtype=torch.float32)
        return input, output
    
# set the lightning model
class PendulumModel(L.LightningModule):
    def __init__(self, sd, id, hd, od):
        super(PendulumModel, self).__init__()
        self.input = torch.nn.Linear(sd+id, hd)
        self.l1 = torch.nn.Linear(hd, hd)
        self.out = torch.nn.Linear(hd, od)
        self.loss = torch.nn.MSELoss()

    def forward(self, x):
        x = torch.relu(self.input(x))
        x = torch.tanh(self.l1(x))
        x = self.out(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss)
        return loss
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
def nn_step(x, u, model):
    with torch.no_grad():
        input = torch.tensor(np.concatenate((x, [u]), axis=0), dtype=torch.float32)
        input = input.unsqueeze(0)
        output = model(input)[0].detach().numpy()
        return output

if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')

    N_EPOCHS = 400
    
    # create the dataset
    tds = Pendulum1StepDataset(DT, N_SAMPLES) # training dataset
    eds = Pendulum1StepDataset(DT, N_SAMPLES//5) # evaluation dataset
    tdl = DataLoader(tds, batch_size=N_SAMPLES, shuffle=True, num_workers=7, persistent_workers=True)
    edl = DataLoader(eds, batch_size=N_SAMPLES//5, shuffle=False, num_workers=7, persistent_workers=True)
    
    # create the model
    model = PendulumModel(sd=2, id=1, hd=128, od=2)
    
    trainer = L.Trainer(max_epochs=N_EPOCHS,
                        log_every_n_steps=1,)
    trainer.fit(model, tdl, edl)

    print('Training completed')

    # save the model
    if not os.path.exists('tmp'): os.makedirs('tmp')
    torch.save(model.state_dict(), 'tmp/pendulum_model.pt')
    
    # test the model
    model.eval()
    x = np.array([0.1, 0])
    u = 0
    x1 = step(x, u, DT)
    x1_nn = nn_step(x, u, model)
    print(f'x1: {x1}, x1_nn: {x1_nn}')
    
    # plot the results
    t = np.linspace(0, SIMT, int(SIMT/DT))
    x = np.zeros((len(t), 2))
    x[0] = np.array([0.1, 0])
    for i in range(1, len(t)): x[i] = step(x[i-1], u, DT)
    x_nn = np.zeros((len(t), 2))
    x_nn[0] = np.array([0.1, 0])
    for i in range(1, len(t)): x_nn[i] = nn_step(x_nn[i-1], u, model)

    # plot the results
    xs = np.array([x, x_nn])
    fig1 = plot_state_trajectories(xs, figsize=(10,10))
    
    # test on an arbitrary dataset
    fig2, ax2 = plt.subplots(1, 1, figsize=(10,10))
    N_GRID = 20
    As, Vs = np.linspace(-π, π, N_GRID), np.linspace(-MAXV, MAXV, N_GRID)
    tds = Pendulum1StepDataset(DT, N_SAMPLES) # training dataset
    for ia, a in enumerate(tqdm(As)):
        for iv, v in enumerate(Vs):
            x0 = np.array([a, v])
            u = 0
            x1 = step(x0, u, DT)
            x1_nn = nn_step(x0, u, model)
            #plot a line from x0 to x1
            if np.linalg.norm(x0-x1_nn) < 0.5:
                ax2.plot([a, x1_nn[0]], [v, x1_nn[1]], 'r')
            if np.linalg.norm(x0-x1) < 0.5:
                ax2.plot([a, x1[0]], [v, x1[1]], 'b')
    ax2.set_title('simulation and neural network prediction')
    ax2.grid(True)
    ax2.set_xlabel('angle')
    ax2.set_ylabel('angular velocity')

    plt.show()
