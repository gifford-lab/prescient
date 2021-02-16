import argparse
import torch
import prescient.simulate as simulate
from prescient.train.model import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--data_path", required=True, help="Path to PRESCIENT data file stored as a torch pt.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--seed", default=1, required=True)
    parser.add_argument("--num_sims", default=10, help="Number of simulations to run.")
    parser.add_argument("--num_cells", default=200, help="Number of cells per simulation.")
    parser.add_argument("--num_steps", default=None, required=False)
    parser.add_argument("--device", default=None, required=False)

    # subsetting parameters
    parser.add_argument("--celltype", default=None, required=False, help="Randomly sample initial cells from a particular celltype defined in metadata")
    parser.add_argument("--tp", default=None, required=False, help="Randomly sample initial cells from a particular timepoint.")

    parser.add_argument("-o", "--path_to_output", default=None)

    # load data
    data_pt = torch.load(args.data_path)

    # load model
    config_path = os.path.join(args.model_path, '/seed_{}/config.pt'.format(args.seed))
    config = SimpleNamespace(**torch.load(config_path))
    train_pt = args.model_path

    model = AutoGenerator(config)
    checkpoint = torch.load(train_pt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Either use assigned number of steps or calculate steps, both using the stepsize used for training
    if args.num_steps == None:
        num_steps = int(np.round(data_pt["y"] / config.train_dt))

    # simulate forward
    sims = simulate(data_pt["xp"], data_pt["y"], data_pt["celltype"], data_pt["w"], model, config, args.num_sims, args.num_cells, num_steps, tp=args.tp, celltype=args.celltype, device=args.device)

    # write simulation data to file
    torch.save({
    "sims": sims
    }, args.path_to_output)

if __name__=="__main__":
    main()
