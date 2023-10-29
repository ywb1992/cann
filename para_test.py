from configs.mylibs import *
from models.model import CANN_Network
from utils.utils import *

CANN = CANN_Network()

def show_para_mat(CANN, core_point):
    print(core_point)
    core_point = CANN.num_index2coor_index[core_point[0], core_point[1]]
    
    
    coor_x = CANN.coor_vec[:, 0]
    coor_y = CANN.coor_vec[:, 1]
    coor_z = CANN.conn_mat[core_point, :]
    coor_x = torch.reshape(coor_x, (CANN.x_len, CANN.y_len))
    coor_y = torch.reshape(coor_y, (CANN.x_len, CANN.y_len))
    coor_z = torch.reshape(coor_z, (CANN.x_len, CANN.y_len))
    
    print(coor_x.size(), coor_y.size(), coor_z.size())
    
    plot_3d(coor_x, coor_y, coor_z)

if __name__ == '__main__':
    core_point = torch.tensor([CANN.x_len // 2, CANN.y_len // 2])
    show_para_mat(CANN, core_point)


exit(0)


# Generate a circle trajectory with 20 points
theta = torch.linspace(0, 2*3.14159, 25)
# circle_track = [(0.5 + 0.2*torch.cos(t), 0.5 + 0.2*torch.sin(t)) for t in theta]
circle_track = [(0.5 + 0.2*torch.cos(t), 0.5 + 0.2*torch.sin(t)) for t in theta]

# Initialize CANN model with PyTorch and an 'a' value of 4 for demonstration
cann_model_torch_full = CANN2D_Torch_Full(a=8)

# Update and record the network state based on the circular track
network_states_torch_full = cann_model_torch_full.update(circle_track)


circle_track_np = np.array(circle_track)
plt.figure(figsize=(10, 10))
for i, state in enumerate(network_states_torch_full):
    plt.subplot(4, 5, i + 1)
    tmp = state.detach().numpy()
    tmp[tmp < tmp.max() * 0.999] = 0.
    
    # Add the objective trajectory point
    x, y = (circle_track_np[i] * (cann_model_torch_full.length - 1)).astype(int)
    tmp[x, y] = tmp.max()  # Set the objective point to the maximum activation value
    
    plt.imshow(tmp, cmap='hot', interpolation='nearest')
    
    # Mark the tracking point in red
    plt.scatter(y, x, color='red')
    
    plt.title(f'Time Step {i}')
plt.tight_layout()
plt.show()
plt.savefig('')