import json

if __name__ == '__main__':
    experiments = []
    for wd in ['-d', None]:
        for cscale in range(2,32):
            experiment = ["python", "dawn.py", "%i"%cscale]
            if wd is not None:
                experiment += [wd]
            experiments.append(experiment)
    with open("experiment_schedule.json", "w") as f:
        f.write(json.dumps(experiments))
