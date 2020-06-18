import sys
import json


if __name__ == '__main__':
    exp = sys.argv[1]
    with open(exp, "r") as f:
        exp = json.load(f)
    model, config = exp["model"], exp["config"]

    if model == "model_template":
        from src.models import model_template
        model_template.run(config)
