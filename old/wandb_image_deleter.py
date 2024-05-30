#!/usr/bin/env python
import ipdb
import wandb
import tqdm
api = wandb.Api()

# sorry wandb api, you will have to suffer to reach the storage limit

# test for one run
# run = api.run('mbjd-projects/levelsets_flatquad/runs/ouum6iv6')
# for file in run.files():
#     if file.name.endswith('png'):
#         print(f'deleting file.name')
#         file.delete()

# now for all of them hehehehe :)

runs = api.runs(path='mbjd-projects/levelsets_flatquad', order='+created_at')

deleted = 0

for run in tqdm.tqdm(runs, position=0):
    for file in (pbar := tqdm.tqdm(run.files(), position=1)):
        if file.name.endswith('png'):
            file.delete()
            deleted += file.size
            pbar.set_description(f'{deleted/1e6} MB')

runs = api.runs(path='mbjd-projects/levelsets_orbits', order='+created_at')

for run in tqdm.tqdm(runs, position=0):
    for file in (pbar := tqdm.tqdm(run.files(), position=1)):
        if file.name.endswith('png'):
            file.delete()
            deleted += file.size
            pbar.set_description(f'{deleted/1e6} MB')


'''
extension = ".png"
files = run.files()
for file in files:
    if file.name.endswith(extension):
        file.delete()
'''
