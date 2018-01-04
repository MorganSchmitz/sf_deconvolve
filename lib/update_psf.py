# -*- coding: utf-8 -*-

"""PSF UPDATE MODULE


"""

from __future__ import print_function
from builtins import range, zip
from scipy.linalg import norm
from sf_tools.signal.optimisation import *
from sf_tools.math.stats import sigma_mad
from sf_tools.signal.linear import *
from sf_tools.signal.proximity import *
from sf_tools.signal.reweight import cwbReweight
from sf_tools.signal.wavelet import filter_convolve, filter_convolve_stack
from gradient import *
from mas_cost_plot import plotCost




def run(data, psf, psf_model, **kwargs):
    """Run deconvolution

    This method initialises the operator classes and runs the optimisation
    algorithm

    Parameters
    ----------
    data : np.ndarray
        Input data array, an array of 2D images
    psf : np.ndarray
        Input PSF array, a single 2D PSF or an array of 2D PSFs

    Returns
    -------
    np.ndarray decconvolved data

    """
    kwargs['grad_op'] = GradUnknownPSF(data, psf, Positive(),
                                           psf_model=psf_model,
                                           psf_type=kwargs['psf_type'],
                                           beta_reg=kwargs['beta_psf'],
                                           beta_sig=kwargs['beta_sig'],
                                           lambda_reg=kwargs['lambda_psf'],
                                           decrease_factor=kwargs['decfac_psf'])
    costs = [kwargs['grad_op'].psf_cost(psf, data)]
    cost = np.sum(costs)
    costs = [costs[0] + (cost,)]
    n_iter = 0
    converged = False
    while not converged:
        print('\n - ITERATION: {}'.format(n_iter))
        kwargs['grad_op'].get_grad(data)
        print(' - Current step size: {}'.format(kwargs['grad_op']._beta_reg))
        upd_psf = kwargs['grad_op']._psf
        this_cost = kwargs['grad_op'].psf_cost(upd_psf, data)
        this_cost += (np.sum(this_cost),)
        costs += [this_cost]
        this_cost = this_cost[-1]
        n_iter += 1
        if np.abs(this_cost - cost) < kwargs['convergence']:
            converged=True
            print('PSF estimation converged! (Cost difference below tolerance)')
        elif (n_iter>kwargs['n_iter']):
            converged=True
            print('PSF estimation converged! (Maximum iterations reached)')
        cost = this_cost
        print(' - PSF COST: {}'.format(cost))
    plotCost(np.array(costs), output=kwargs['output'], psf_only=True)
    
    return data, None, upd_psf
