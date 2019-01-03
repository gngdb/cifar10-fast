import json

if __name__ == '__main__':
    experiments = []
    for rankscale in range(1,20):
        rankscale = rankscale/20.
        for dimensions in range(2,5):
            experiment = ["python", "dawn.py", "%f"%rankscale, "%i"%dimensions, "tt"]
            experiments.append(experiment)
    for rankscale in range(1,10):
        rankscale = rankscale/10.
        for dimensions in range(2,10):
            experiment = ["python", "dawn.py", "%f"%rankscale, "%i"%dimensions, "tucker"]
            experiments.append(experiment)
    print("Number of experiments: ", len(experiments))
    print("Time to run: %i minutes"%(len(experiments)*20))
    print("Time to run on 8 GPUs: %i minutes"%(len(experiments)*20/8))
    with open("experiment_schedule.json", "w") as f:
        f.write(json.dumps(experiments))
