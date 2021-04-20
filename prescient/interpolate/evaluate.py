

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
    ot_solver = SamplesLoss("sinkhorn", p = 2, blur = config.sinkhorn_blur,
        scaling = config.sinkhorn_scaling)
    distance = ot_solver(x_i, y_j)
    return distance

def evaluate_interpolate_data(y_j, data_pt_path, simulate_pt_path, num_steps):
    """
    Approximates W-2 distance between sets of points using Sinkhorn as implemented in GeomLoss.

    Arguments
    ---------
    - y_j: held-out timepoint (as Numpy ndarray normalized expression).
    - data_pt_path: path to serialized torch object produced by "process_data" cmd.
    - simulate_pt_path: path to serialized torch object produced by "simulate_trajectiories" cmd.

    Returns:
    --------
    loss_xy: W-2 distance between observed and simulated points at the heldout timepoint.
    """

    # load PRESCIENT outputs
    data_pt = torch.load(dat_pt_path)
    simulate_pt = torch.load(simulate_pt_path)

    # transform held-out timepoint using PCA from data_pt
    pca = data_pt["pca"]
    y_j_ = pca.transform(y_j)
    y_j = x[1].to(device)

    # evaluate interpolation distance for each simulation in simulate_pt
    sims = simulate_pt["sims"]
    interpolate_distances=[]
    for sim in sims:
        x_i = torch.from_numpy(sim[:, :, :, num_steps]).float().to(device)
        interpolate_distances.append(w2dist(x_i, y_j))
    return interpolate_distances



    return interpolate_error

def evaluate_interpolate_models():
    """
    Wrapper to evaluate multiple models at once.
    """
    pass
