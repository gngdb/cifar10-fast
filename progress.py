import json

def not_run_yet(schedule):
    return [i for i,e in enumerate(schedule) if e[0] != "executed"]

def progress():
    running_json = 'experiment_schedule.json'
    with open(running_json, "r") as f:
        schedule = json.load(f)
    # filter for experiments that have not been run yet
    done = len(schedule) - len(not_run_yet(schedule))
    print("Progress: %.2f%% (%i/%i)"%(100*float(done)/float(len(schedule)),
        done, len(schedule)))

if __name__ == '__main__':
    progress()
