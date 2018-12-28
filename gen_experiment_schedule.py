import json

if __name__ == '__main__':
    experiments = []
    for rankscale in range(2,16):
        for dimensions in range(2,16):
            experiment = ["python", "dawn.py", "%i"%rankscale, "%i"%dimensions]
            experiments.append(experiment)
    print("Number of experiments: ", len(experiments))
    print("Time to run: %i minutes"%(len(experiments)*20))
    print("Time to run on 8 GPUs: %i minutes"%(len(experiments)*20/8))
    with open("experiment_schedule.json", "w") as f:
        f.write(json.dumps(experiments))
