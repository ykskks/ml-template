import sys
import json

sys.path.append(".")


if __name__ == '__main__':
    exp = sys.argv[1]
    # TODO: make the extraction of exp_id better
    exp_id = exp[-11:-5]
    with open(exp, "r") as f:
        exp = json.load(f)
    model, config = exp["model"], exp["config"]

    if model == "lgbm":
        from src.models.lgbm import Experiment
        Experiment(exp_id, config, "lgbm").run()
