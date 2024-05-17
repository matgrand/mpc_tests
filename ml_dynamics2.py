import torch, os, numpy as np, matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import lightning as L
from time import time, sleep
from single_pendulum import *
from plotting import *

torch.set_float32_matmul_precision('medium')

LOAD_PRETRAIN = True

MAXV = 8 # maximum velocity
MAXU = 4 # maximum control input
DT = 1/60 # time step 
NT_SAMPLES = 5_000_000 # number of training samples
NV_SAMPLES = NT_SAMPLES//3 # number of validation samples
N_EPOCHS = 200 # number of epochs
WORKERS = 16 # macos: 10, ubuntu: 15, cluster: check job.sh
BATCH_SIZE = NT_SAMPLES // WORKERS # batch size
LR = 1e-4 # learning rate

SIMT = 15 # simulation time

WRAP_AROUND = False # wrap around the angle to [-π, π]

def create_dataset(n, dt, file_path='pendulum_dataset.npz'):
    θ0s = np.random.uniform(-2*π, 2*π, n).astype(np.float32)
    dθ0s = np.random.uniform(-MAXV, MAXV, n).astype(np.float32)
    # us = np.random.uniform(-MAXU, MAXU, n).astype(np.float32)
    us = np.zeros(n, dtype=np.float32)
    x0s = np.vstack((θ0s, dθ0s)).T # initial states
    x1s = np.zeros((n, 2), dtype=np.float32) # final states
    assert x0s.shape == x1s.shape, f'x0s: {x0s.shape}, x1s: {x1s.shape}'
    assert x0s.dtype == x1s.dtype == np.float32, f'x0s: {x0s.dtype}, x1s: {x1s.dtype}'
    for i in tqdm(range(n)): 
        x1s[i] = step(x0s[i], us[i], dt, wa=WRAP_AROUND) # simulate the pendulum
    np.savez(file_path, x0s=x0s, us=us, x1s=x1s, dt=dt, n=n) # save the dataset in a file

# torch dataset
class Pendulum1StepDataset(Dataset):
    def __init__(self, file_path):
        super(Pendulum1StepDataset, self).__init__()
        data = np.load(file_path)
        x0s = torch.tensor(data['x0s'], dtype=torch.float32)
        us = torch.tensor(data['us'], dtype=torch.float32)
        self.x0s = torch.cat((x0s, us.unsqueeze(1)), dim=1) # concatenate the states and control inputs
        self.x1s = torch.tensor(data['x1s'], dtype=torch.float32)
        self.n = len(self.x0s)
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        return self.x0s[idx], self.x1s[idx]
    
# set the lightning model
class PendulumModel(L.LightningModule):
    def __init__(self, sd, id, hd, od):
        super(PendulumModel, self).__init__()
        self.input = torch.nn.Linear(sd+id, hd)
        self.l1 = torch.nn.Linear(hd, hd)
        self.l2 = torch.nn.Linear(hd, hd)
        self.out = torch.nn.Linear(hd, od)
        self.loss = torch.nn.MSELoss()
        self.best_val_loss = np.inf

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

        if loss < self.best_val_loss:
            self.best_val_loss = loss
            torch.save(self.state_dict(), 'tmp/best_pendulum_model.pt')

        return loss
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)
        # return torch.optim.SGD(self.parameters(), lr=LR*50, momentum=0.9)
    
def nn_step(x, u, model):
    with torch.no_grad():
        input = torch.tensor(np.concatenate((x, [u]), axis=0), dtype=torch.float32)
        return model(input.unsqueeze(0)).squeeze(0).detach().numpy()

if __name__ == '__main__':
    # os.system('rm -rf lightning_logs') # remove the logs

    start_time = time()

    # create directories
    if not os.path.exists('ds'): os.makedirs('ds') # create the directory to save the dataset
    if not os.path.exists('tmp'): os.makedirs('tmp') # create the directory to save the model

    # datasets & dataloaders
    tds_path = f'ds/train_ds_pendulum_{NT_SAMPLES}.npz'
    vds_path = f'ds/val_ds_pendulum_{NV_SAMPLES}.npz'
    if not os.path.exists(tds_path): create_dataset(NT_SAMPLES, DT, tds_path)
    if not os.path.exists(vds_path): create_dataset(NV_SAMPLES, DT, vds_path)
    tds = Pendulum1StepDataset(tds_path) # training dataset
    eds = Pendulum1StepDataset(vds_path) # validation dataset
    tdl = DataLoader(tds, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, persistent_workers=True)
    edl = DataLoader(eds, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, persistent_workers=True)
    
    # create the model
    model = PendulumModel(sd=2, id=1, hd=400, od=2)

    if not LOAD_PRETRAIN:
        trainer = L.Trainer(max_epochs=N_EPOCHS,
                            log_every_n_steps=1,)
        trainer.fit(model, tdl, edl)
        print('Training completed')
        # save the model
        torch.save(model.state_dict(), 'tmp/last_pendulum_model.pt')

    model.load_state_dict(torch.load('tmp/best_pendulum_model.pt'))
    print('Model loaded')
        
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
    Anorms = np.zeros((N_GRID, N_GRID))
    Vnorms = np.zeros((N_GRID, N_GRID))
    for ia, a in enumerate(tqdm(As)):
        for iv, v in enumerate(Vs):
            x0 = np.array([a, v])
            u = 0
            # x1 = step(x0, u, DT, wa=WRAP_AROUND)
            x1 = step(x0, u, DT, wa=True)
            x1_nn = nn_step(x0, u, model)
            Anorms[ia, iv] = np.linalg.norm(x1[0]-x1_nn[0])
            Vnorms[ia, iv] = np.linalg.norm(x1[1]-x1_nn[1])
            #plot a line from x0 to x1
            if np.linalg.norm(x0-x1_nn) < 0.5:
                ax2.plot([a, x1_nn[0]], [v, x1_nn[1]], 'r')
            if np.linalg.norm(x0-x1) < 0.5:
                ax2.plot([a, x1[0]], [v, x1[1]], 'b')
    ax2.set_title('simulation and neural network prediction')
    ax2.grid(True)
    ax2.set_xlabel('angle')
    ax2.set_ylabel('angular velocity')
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(20,10))
    c1 = ax3a.contourf(As, Vs, Anorms.T, levels=50)
    fig3.colorbar(c1, ax=ax3a)
    ax3a.set_title('norm of the difference in angle')
    ax3a.set_xlabel('angle')
    ax3a.set_ylabel('angular velocity')

    c2 = ax3b.contourf(As, Vs, Vnorms.T, levels=50)
    fig3.colorbar(c2, ax=ax3b)
    ax3b.set_title('norm of the difference in angular velocity')
    ax3b.set_xlabel('angle')
    ax3b.set_ylabel('angular velocity')

    print(f'Anorms: avg: {np.mean(Anorms):.5f}, std: {np.std(Anorms):.5f}, min: {np.min(Anorms):.5f}, max: {np.max(Anorms):.5f}')
    print(f'Vnorms: avg: {np.mean(Vnorms):.5f}, std: {np.std(Vnorms):.5f}, min: {np.min(Vnorms):.5f}, max: {np.max(Vnorms):.5f}')

    plt.show()
    #save the plots in logs
    if not os.path.exists('logs'): os.makedirs('logs')
    anim.save('logs/pendulum.gif', writer='imagemagick', fps=30)
    fig1.savefig('logs/pendulum_trajectories.png')
    fig2.savefig('logs/pendulum_comparison.png')

    print(f'Execution time: {(time()-start_time)/60:.2f} mins')
    print('Done')
