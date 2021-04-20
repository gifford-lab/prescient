

from geomloss import SamplesLoss

def w2dist(x, y):
    """
    Approximates W-2 distance between sets of points using Sinkhorn as implemented in GeomLoss.
    Arguments
    ---------
    - x: first set of points.
    - y: second set of points.

    Returns:
    --------
    loss_xy: W-2 distance between observed and simulated points at the heldout timepoint.
    """


def evaluate_interpolate_data(y, data_pt_path, simulate_pt_path, num_steps, gpu=None):
    """
    Approximates W-2 distance between sets of points using Sinkhorn as implemented in GeomLoss.

    Arguments
    ---------
    - y -> numpy.ndarray: held-out timepoint (as Numpy ndarray normalized expression).
    - data_pt_path -> str: path to serialized torch object produced by "process_data" cmd.
    - simulate_pt_path -> str: path to serialized torch object produced by "simulate_trajectiories" cmd.
    - num_steps -> int: number of steps taken with "simulate_trajectories". (i.e the number of steps to get to the held-out timepoint)
    - gpu -> int: if CUDA-enabled gpu enabled provide GPU number (i.e if cuda:0 is availale, provide 0) -> int
    Returns:
    --------
    interpolation_distances -> list(torch.Tensors): W-2 distance between observed and simulated points at the heldout timepoint.
    """

    if gpu != None:
        device = torch.device('cuda:{}'.format(gpu))
    else:
        device = torch.device('cpu')

    # load PRESCIENT outputs
    data_pt = torch.load(data_pt_path)
    simulate_pt = torch.load(simulate_pt_path)

    # transform held-out timepoint using PCA from data_pt
    pca = data_pt["pca"]
    y_ = torch.from_numpy(pca.transform(y)).float().to(device)

    # evaluate interpolation distance for each simulation in simulate_pt
    sims = simulate_pt["sims"]
    interpolate_distances=[]
    ot_solver = SamplesLoss("sinkhorn", p = 2, blur = config.sinkhorn_blur,
        scaling = config.sinkhorn_scaling)
    return distance
    for sim in sims:
        x_i = torch.from_numpy(sim[num_steps,:,:]).float().to(device)
        interpolate_distances.append(ot_solver(x_i, y_))
    return interpolate_distances


def evaluate_interpolate_models():
    """
    Wrapper to evaluate multiple models at once.
    """
    pass
