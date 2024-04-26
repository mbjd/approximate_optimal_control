import aim
import math

# Initialize a new run
run = aim.Run()

# Log hyper-parameters
run["hparams"] = {
    "learning_rate": 0.001,
    "batch_size": 32,
}

# Log metrics
for step in range(100):
    run.track(math.sin(step), name='sine')
    run.track(math.cos(step), name='cosine')