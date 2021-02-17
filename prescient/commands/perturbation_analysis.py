import argparse
import prescient.simulate as traj
from prescient.train.model import *
import prescient.perturb as pert
import prescient.simulate as traj


def create_parser():
    parser = argparse.ArgumentParser()

    # perturbation parameters
    parser.add_argument("-p", "--perturb_genes", required=True, help="Provide a gene or list of genes to be perturbed as a string (commas, no spaces).")
    parser.add_argument("-z", "--z_score", default=5.0, required=True, help="Set magnitude of perturbation as z-score.")

    # simulation parameters
    parser.add_argument("-i", "--data_path", required=True, help="Path to PRESCIENT data file stored as a torch pt.")
    parser.add_argument("--model_path", required=True, help="Path to model directory.")
    parser.add_argument("--seed", default=1, required=True, help="Choose the seed of the trained model to use for simulations.")
    parser.add_argument("--epoch", default="002500", type=str, required=False, help="Choose which epoch of the model to use for simulations.")
    parser.add_argument("--num_sims", default=10, help="Number of simulations to run.")
    parser.add_argument("--num_cells", default=200, help="Number of cells per simulation.")
    parser.add_argument("--num_steps", default=None, required=False, help="Define number of forward steps of size dt to take.")
    parser.add_argument("--gpu", default=None, required=False)
    parser.add_argument("--celltype_subset", default=None, required=False, help="Randomly sample initial cells from a particular celltype defined in metadata.")
    parser.add_argument("--tp_subset", default=None, required=False, help="Randomly sample initial cells from a particular timepoint.")
    parser.add_argument("-o", "--out_path", required=True, default=None, help="Path to output directory.")
    return parser


def main(args):

    # load data
    data_pt = torch.load(args.data_path)
    genes = data_pt["genes"].tolist()
    expr = data_pt["data"]
    pca = data_pt["pca"]
    xp = pca.transform(expr)
    xp = xp[:,0:30]

    # generate perturbations PRESCIENT data file
    xp_perturb = pert.z_score_perturbation(genes, args.perturb_genes, expr, pca, args.z_score)
    xp_perturb = xp_perturb[:,0:30]
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
        t = data_pt["y"][-1]-data_pt["y"][0]
        num_steps = int(np.round(t / config.train_dt))
    else:
        num_steps = int(args.num_steps)

    # simulate forward
    std_out = traj.simulate(xp, data_pt["tps"], data_pt["celltype"], data_pt["w"], net, config, args.num_sims, args.num_cells, num_steps, device, args.tp_subset, args.celltype_subset)

    perturbed_out = traj.simulate(xp_perturb, data_pt["tps"], data_pt["celltype"], data_pt["w"], net, config, args.num_sims, args.num_cells, num_steps, device, args.tp_subset, args.celltype_subset)

    out_path = os.path.join(args.out_path, args.model_path.split("/")[-1], 'seed_{}_train.epoch_{}_num.sims_{}_num.cells_{}_num.steps_{}_subsets_{}_{}_perturb_simulation.pt'.format(args.seed, args.epoch, args.num_sims, args.num_cells, num_steps, args.tp_subset, args.celltype_subset))
    # save PRESCIENT perturbation file
    torch.save({"perturbed_genes": args.perturb_genes,
                "unperturbed_sim": std_out,
                "perturbed_sim": perturbed_out},
               out_path)


if __name__=="__main__":
    main()
