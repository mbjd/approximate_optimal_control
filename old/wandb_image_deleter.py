#!/usr/bin/env python
import ipdb
import tqdm

import wandb

api = wandb.Api()

# sorry wandb api, you will have to suffer to reach the storage limit

# test for one run
# run = api.run('mbjd-projects/levelsets_flatquad/runs/ouum6iv6')
# for file in run.files():
#     if file.name.endswith('png'):
#         print(f'deleting file.name')
#         file.delete()

# now for all of them hehehehe :)



def delete_pngs(runs):
    deleted = 0
    for run in (pbar := tqdm.tqdm(runs)):
        for file in run.files():
            if file.name.endswith('png'):
                file.delete()
                deleted += file.size
            pbar.set_description(f'{deleted/1e6:.2f} MB')

runs = api.runs(path='mbjd-projects/levelsets_flatquad', order='+created_at')
delete_pngs(runs)

runs = api.runs(path='mbjd-projects/levelsets_orbits', order='+created_at')
delete_pngs(runs)

'''
extension = ".png"
files = run.files()
for file in files:
    if file.name.endswith(extension):
        file.delete()
'''
