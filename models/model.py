import torch


# Define a forward function to replace ODEint
def forward(u, conn_mat, cur_stimulus, tau, Inh_inp, k, dt, steps):
    u_record = []
    for _ in range(steps):
        r1 = torch.square(u)
        r2 = 1.0 + k * torch.sum(r1)
        r = r1 / r2
        Irec = torch.matmul(r, conn_mat)
        du = (-u + Irec + cur_stimulus + Inh_inp) * dt / tau
        u = u + du
        u_record.append(u)
    return torch.stack(u_record)

# Modified CANN2D class to use PyTorch and remove NumPy dependencies
class CANN_Network():
    def __init__(self, x_len=100, y_len=100, tau=0.02, Inh_inp=-0.1, I_dur=0.4, dt=0.02, A=2, k=1, a=1):
        self.x_len = x_len
        self.y_len = y_len
        self.num = x_len * y_len
        self.x_range = 2 * torch.pi
        self.y_range = 2 * torch.pi

        self.tau = tau
        self.Inh_inp = Inh_inp #
        self.I_dur = I_dur #
        self.dt = dt
        self.A = A
        self.k = k
        self.a = a
        
        self.x = torch.linspace(0, self.x_range, self.x_len)
        self.y = torch.linspace(0, self.y_range, self.y_len)
        self.u = torch.zeros((self.x_len, self.y_len))
        self.coor_vec = torch.zeros((self.num, 2)) # N * 2
        self.num_index2coor_index = torch.reshape(torch.arange(0, self.num, 1),
                                                  (self.x_len, self.y_len)) # x_len * y_len
        self.dist_mat = torch.zeros((self.num, self.num)) # N * N
        self.conn_mat = torch.zeros((self.num, self.num)) # N * N
        self.get_deduced_para()

    def get_dist(self, delta_pos):
        """
        Args:
            delta_pos (torch.tensor([x, y])): delta coordinate

        Returns:
            (torch.tensor([x, y])): refined delta coordinate
        """
        xy = torch.as_tensor([self.z_range, self.z_range])
        refined_delta_pos = torch.where(delta_pos > xy / 2, xy - delta_pos, delta_pos)
        return torch.norm(refined_delta_pos)
        
    def get_deduced_para(self):
        """
        Functions:
            Get the self.coor_ver, self.dist_mat, self.conn_mat
        States:
            Testing.
        """
        x, y = torch.meshgrid(self.x, self.y) # To generate grid points, x1 and x2 are 2D tensors
        self.coor_vec = torch.stack([x.flatten(), y.flatten()]).T # Flatten to row vectors, then stack and transpose
        coor_extend = self.coor_vec[:, :, None] # (N, 2, 1)
        coor_extend_T = coor_extend.transpose(0, 2) # (1, 2, N)
        delta_coor = coor_extend - coor_extend_T # (N, 2, N)
        delta_coor = torch.abs(delta_coor) # to get the absolute delta
        
        # next step: For the circular coordinate, we need to jusify the distance
        delta_coor[:, 0, :] = torch.where(delta_coor[:, 0, :] > self.x_range / 2,
                                          self.x_range - delta_coor[:, 0, :], delta_coor[:, 0, :])
        delta_coor[:, 1, :] = torch.where(delta_coor[:, 1, :] > self.y_range / 2,
                                          self.y_range - delta_coor[:, 1, :], delta_coor[:, 1, :])
        
        self.dist_mat = torch.norm(delta_coor, dim=1) # (N, N)
        self.conn_mat = self.A * (torch.exp(-0.5 * torch.square(self.dist_mat / self.a)) / 
                            (torch.sqrt(2 * torch.tensor(torch.pi)) * self.a)
                       )

    def get_stimulus_by_pos(self, x):
        return self.A * torch.exp(-0.25 * torch.square(self.dist_mat[x, :] / 2))

    def update(self, data):
        seq_len = len(data)
        u = torch.zeros((self.x_len, self.y_len))
        u_record = torch.zeros((seq_len, self.x_len, self.y_len))
        steps = int(self.I_dur / self.dt)
        for i in range(1, seq_len):
            cur_stimulus = self.get_stimulus_by_pos(data[i - 1])
            cur_du = forward(u, self.conn_mat, cur_stimulus, self.tau, self.Inh_inp, self.k, self.dt, steps)
            u = cur_du[-1]
            r1 = torch.square(u)
            r2 = 1.0 + self.k * torch.sum(r1)
            u_record[i] = r1/r2
        return u_record

