import argparse
import prescient.simulate as simulate
from prescient.train.model import *
import prescient.perturb


def main():
    parser = argparse.ArgumentParser()

    # perturbation parameters
    parser.add_argument("-p", "--perturb_genes", required=True, help="Provide a gene or list of genes to be perturbed (commas, no spaces).")
    parser.add_argument("-z", "--z_score", default=5.0, required=True, help="Set magnitude of perturbation as z-score.")

    # simulation parameters
    parser.add_argument("-i", "--data_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--seed", default=1, required=True)
    parser.add_argument("--num_sims", default=10)
    parser.add_argument("--num_steps", default=None, required=False)
    parser.add_argument("--device", default=None, required=False)
    parser.add_argument("--out_path", default=None, required=True)

    # subsetting parameters
    parser.add_argument("--celltype", default=None, required=False, help="Randomly sample initial cells from a particular celltype defined in metadata")
    parser.add_argument("--tp", default=None, required=False, help="Randomly sample initial cells from a particular timepoint.")

    # load data
    data_pt = torch.load(args.data_path)
    x = data_pt["x"]
    xp = data_pt["xp"]
    y = data_pt["y"]
    celltype_annotation = data_pt["celltype"]
    w = data_pt["w"]

    # load model
    config_path = os.path.join(args.model_path, '/seed_{}/config.pt'.format(args.seed))
    config = SimpleNamespace(**torch.load(config_path))
    train_pt = args.model_path

    model = AutoGenerator(config)
    checkpoint = torch.load(train_pt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # generate perturbations PRESCIENT data file
    xp_perturb = prescient.perturb.z_score_perturbation(args.perturb_genes, data_pt["x"], pca, z_score)

    # save perturbed data_pt in same location as

    # Either use assigned number of steps or calculate steps, both using the stepsize used for training
    if args.num_steps == None:
        t = data_pt["y"][-1]-data_pt["y"][0]
        num_steps = int(np.round(t / config.train_dt))

    # simulate forward
    sims = simulate(xp, y, celltype_annotation, w, model, config, num_sims, num_cells, num_steps, tp=args.tp, celltype=args.celltype, gpu=args.device)
    perturbed_sims = simulate(xp_perturb, y, celltype_annotation, perturb_pt, model, config, num_sims, num_cells, tp=args.tp, celltype=args.celltype, gpu=args.device)

    # save PRESCIENT perturbation file
    torch.save({"perturbed_genes": args.perturb_genes,
                "unperturbed_sim": unperturbed_sims,
                "unperturbed_labs": unperturbed_classes,
                "perturbed_sim": perturbed_sims,
                "perturbed_labs": perturbed_classes},
               args.out_path)


if __name__=="__main__":
    main()
