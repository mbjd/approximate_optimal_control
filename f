
mus, sigmas = v_meanstds(xs, vmap_params)

ax = pl.subplot(211)
pl.plot(thetas, mus, label='value mean')
pl.fill_between(thetas, mus - sigmas, mus + sigmas, color='C0', alpha=.2, label=f'value 1σ confidence')
pl.legend()

vx_mu, vx_sigma = vx_meanstds(xs, vmap_params)

pl.subplot(212, sharex=ax)
pl.plot(thetas, vx_mu, label=problem_params['state_names'])

pl.gca().set_prop_cycle(None)

for j in range(7):
    pl.fill_between(thetas, vx_mu[:, j] - vx_sigma[:, j], vx_mu[:, j] + vx_sigma[:, j], alpha=.2)

pl.legend()
