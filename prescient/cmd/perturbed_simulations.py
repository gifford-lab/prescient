import argparse
import prescient.simulate as simulate
from prescient.train.model import *


def main():
    parser = argparse.ArgumentParser()

    # perturbation parameters

    # simulation parameters
    parser.add_argument("-i", "--data_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--seed", default=1, required=True)
    parser.add_argument("--num_sims", default=10)
    parser.add_argument("--num_steps", default=None, required=False)


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

    # generate perturbations PRESCIENT data file
    

    # Either use assigned number of steps or calculate steps, both using the stepsize used for training
    if args.num_steps == None:
        num_steps = int(np.round(data_pt["y"] / config.train_dt))

    # simulate forward
    sims = simulate(data_pt, model, config, num_sims, num_cells, num_steps, tp=None, celltype=None, gpu=None)

    torch.save({"perturbed_genes": args.perturb_genes,
                "unperturbed_sim": unperturbed_sims,
                "unperturbed_labs": unperturbed_classes,
                "perturbed_sim": perturbed_sims,
                "perturbed_labs": perturbed_classes},
               outfile)


if __name__=="__main__":
    main()
