import argparse
import torch
import prescient.simulate as traj
from prescient.train.model import *

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--data_path", required=True, help="Path to PRESCIENT data file stored as a torch pt.")
    parser.add_argument("--model_path", required=True, help="Path to directory containing PRESCIENT model for simulation.")
    parser.add_argument("--seed", default=1, required=True, help="Choose the seed of the trained model to use for simulations.")
    parser.add_argument("--epoch", default="002500", type=str, required=False, help="Choose which epoch of the model to use for simulations.")
    parser.add_argument("--num_sims", default=10, help="Number of simulations to run.")
    parser.add_argument("--num_cells", default=200, help="Number of cells per simulation.")
    parser.add_argument("--num_steps", default=None, required=False, help="Define number of forward steps of size dt to take.")
    parser.add_argument("--gpu", default=None, required=False, help="If available, assign GPU device number.")
    parser.add_argument("--celltype_subset", default=None, required=False, help="Randomly sample initial cells from a particular celltype defined in metadata.")
    parser.add_argument("--tp_subset", default=None, required=False, help="Randomly sample initial cells from a particular timepoint.")
    parser.add_argument("-o", "--out_path", required=True, default=None, help="Path to output directory.")
    return parser

def main(args):

    # load data
    data_pt = torch.load(args.data_path)
    expr = data_pt["data"]
    pca = data_pt["pca"]
    xp = pca.transform(expr)

    # torch device
    if args.gpu != None:
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')

    # load model
    config_path = os.path.join(str(args.model_path), 'seed_{}/config.pt'.format(args.seed))
    config = SimpleNamespace(**torch.load(config_path))
    net = AutoGenerator(config)

    train_pt = os.path.join(args.model_path, 'seed_{}/train.epoch_{}.pt'.format(args.seed, args.epoch))
    checkpoint = torch.load(train_pt, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])

    net.to(device)

    # Either use assigned number of steps or calculate steps, both using the stepsize used for training
    if args.num_steps == None:
        num_steps = int(np.round(data_pt["y"] / config.train_dt))
    else:
        num_steps = int(args.num_steps)

    # simulate forward
    out = traj.simulate(xp, data_pt["tps"], data_pt["celltype"], data_pt["w"], net, config, args.num_sims, int(args.num_cells), num_steps, device, args.tp_subset, args.celltype_subset)

    # write simulation data to file
    out_path = os.path.join(args.out_path, args.model_path.split("/")[-1], 'seed_{}_train.epoch_{}_num.sims_{}_num.cells_{}_num.steps_{}_subsets_{}_{}_simulation.pt'.format(args.seed, args.epoch, args.num_sims, args.num_cells, num_steps, args.tp_subset, args.celltype_subset))
    torch.save({
    "sims": out
    }, out_path)

if __name__=="__main__":
    main()
