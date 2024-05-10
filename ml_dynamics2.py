import torch, os, numpy as np, matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import lightning as L
from time import time, sleep
from single_pendulum import *
from plotting import *

LOAD_PRETRAIN = False

MAXV = 8 # maximum velocity
MAXU = 4 # maximum control input
DT = 1/60 # time step 
NT_SAMPLES = 5_000_000 # number of training samples
NV_SAMPLES = NT_SAMPLES//3 # number of validation samples
N_EPOCHS = 400 # number of epochs
WORKERS = 16 # macos: 7, ubuntu: 15
BATCH_SIZE = NT_SAMPLES // WORKERS # batch size
LR = 1e-4 # learning rate

SIMT = 15 # simulation time

WRAP_AROUND = False # wrap around the angle to [-π, π]

def to_torch(x: np.ndarray) -> torch.tensor:
    return 0
    
# torch dataset
class Pendulum1StepDataset(Dataset):
    def __init__(self, dt, n):
        self.dt = dt
        self.n = n

    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        θ0, dθ0, u = np.random.uniform(-π, π), np.random.uniform(-MAXV, MAXV), np.random.uniform(-MAXU, MAXU)
        x1 = step(np.array([θ0, dθ0]), u, self.dt, wa=WRAP_AROUND)
        input = torch.tensor(np.array([θ0, dθ0, u]), dtype=torch.float32)
        output = torch.tensor(x1, dtype=torch.float32)
        return input, output
    
# set the lightning model
class PendulumModel(L.LightningModule):
    def __init__(self, sd, id, hd, od):
        super(PendulumModel, self).__init__()
        self.input = torch.nn.Linear(sd+id, hd)
        self.l1 = torch.nn.Linear(hd, hd)
        self.l2 = torch.nn.Linear(hd, hd)
        self.out = torch.nn.Linear(hd, od)
        self.loss = torch.nn.MSELoss()

    def forward(self, x):
        x = torch.relu(self.input(x))
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
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
        return torch.optim.Adam(self.parameters(), lr=LR)
        # return torch.optim.SGD(self.parameters(), lr=LR*50, momentum=0.9)
    
def nn_step(x, u, model):
    with torch.no_grad():
        input = torch.tensor(np.concatenate((x, [u]), axis=0), dtype=torch.float32)
        return model(input.unsqueeze(0)).squeeze(0).detach().numpy()

if __name__ == '__main__':
    # os.system('cls' if os.name == 'nt' else 'clear')
    # os.system('rm -rf lightning_logs') # remove the logs

    #start tensorboard
    # os.system('tensorboard --logdir=lightning_logs/ --port=6006 --bind_all &')

    start_time = time()

    # create the dataset
    tds = Pendulum1StepDataset(DT, NT_SAMPLES) # training dataset
    eds = Pendulum1StepDataset(DT, NV_SAMPLES) # evaluation dataset
    tdl = DataLoader(tds, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, persistent_workers=True)
    edl = DataLoader(eds, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, persistent_workers=True)
    
    # create the model
    model = PendulumModel(sd=2, id=1, hd=200, od=2)

    if LOAD_PRETRAIN:
        model.load_state_dict(torch.load('tmp/pendulum_model1.pt'))
        print('Model loaded')
    else:
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
    x1 = step(x, u, DT, wa=WRAP_AROUND)
    x1_nn = nn_step(x, u, model)
    print(f'x1: {x1}, x1_nn: {x1_nn}')
    
    # create trajectories
    t = np.linspace(0, SIMT, int(SIMT/DT))
    x = np.zeros((len(t), 2))
    x0 = np.array([np.random.uniform(-π, π), np.random.uniform(-MAXV, MAXV)])
    x[0] = x0
    for i in range(1, len(t)): x[i] = step(x[i-1], u, DT, wa=WRAP_AROUND)
    x_nn = np.zeros((len(t), 2))
    x_nn[0] = x0
    for i in range(1, len(t)): x_nn[i] = nn_step(x_nn[i-1], u, model)

    #animate the pendulum
    xs, us = np.array([x, x_nn]), np.zeros((2, len(t)))
    anim = animate_pendulums(xs, us, t[1]-t[0], l=1, fps=60, figsize=(10,10))

    # plot the results
    xs = np.array([x, x_nn])
    fig1 = plot_state_trajectories(xs, figsize=(10,10))
    
    # test on an arbitrary dataset
    fig2, ax2 = plt.subplots(1, 1, figsize=(10,10))
    N_GRID = 50
    As, Vs = np.linspace(-π, π, N_GRID), np.linspace(-MAXV, MAXV, N_GRID)
    tds = Pendulum1StepDataset(DT, NT_SAMPLES) # training dataset
    for ia, a in enumerate(tqdm(As)):
        for iv, v in enumerate(Vs):
            x0 = np.array([a, v])
            u = 0
            x1 = step(x0, u, DT, wa=WRAP_AROUND)
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

    # plt.show()
    #save the plots in logs
    if not os.path.exists('logs'): os.makedirs('logs')
    anim.save('logs/pendulum.gif', writer='imagemagick', fps=30)
    fig1.savefig('logs/pendulum_trajectories.png')
    fig2.savefig('logs/pendulum_comparison.png')

    print(f'Execution time: {(time()-start_time)/60:.2f} mins')
    print('Done')
