# open experiment_schedule.json, remove one entry and run it
import json
import subprocess

if __name__ == '__main__':
    try:
        while True:
            # read all experiments
            with open("experiment_schedule.json", "r") as f:
                schedule = json.load(f)
            # remove one
            experiment_to_run = schedule.pop()
            # write experiments back with this one removed
            with open("experiment_schedule.json", "w") as f:
                f.write(json.dumps(schedule))
            print("Running: ", " ".join(experiment_to_run))
            subprocess.run(experiment_to_run)
    except IndexError:
        print("All experiments completed.")
