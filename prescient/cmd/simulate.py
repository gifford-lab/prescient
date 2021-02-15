import argparse
import torch
import prescient.simulate as simulate
from prescient.train.model import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--data_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--seed", default=1, required=True)
    parser.add_argument("--num_sims", default=10)
    parser.add_argument("--num_cells", default=200, help="Number of cells per simulation.")
    parser.add_argument("--num_steps", default=None, required=False)
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
    sims = simulate(data_pt, model, config, args.num_sims, args.num_cells, num_steps, tp=None, celltype=None, gpu=None)

    # write simulation data to file
    torch.save({
    "sims": sims
    }, args.path_to_output)

if __name__=="__main__":
    main()
