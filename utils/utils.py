import matplotlib.pyplot as plt


def plot_3d(X, Y, Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    fig.colorbar(surf)  
    plt.show()


