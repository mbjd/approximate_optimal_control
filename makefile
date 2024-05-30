# what would be really cool:
# define target like orbit_plots_RUNID for variable run id, and then we can
# say make orbit_plots_$ID. and the steps to make it would probably be:
# ./experiment.py --eval=blabla/$ID
# ./plot.py (somehow pass it the id as well)
# or keep the IDs hardcoded somewhere but at least in one place?

plots: orbits_plots

orbits_plots: orbits_figure.py fig_config.py
	./orbits_figure.py
