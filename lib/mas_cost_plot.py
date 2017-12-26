# -*- coding: utf-8 -*-

"""PLOTTING ROUTINES

This module contains methods for making plots.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.1

:Date: 05/01/2017

"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt


def plotCost(cost_list, output=None):
    """Plot cost function

    Plot the final cost function

    Parameters
    ----------
    cost_list : list
        List of cost function values
    output : str, optional
        Output file name

    """

    if isinstance(output, type(None)):
        file_name = 'cost_function'

    else:
        file_name = output + '_cost_function'

    nb_cost = cost_list.shape[1]
    if nb_cost == 4:
        labels = ['Data Fid', 'Sparsity', 'PSF update', 'Full']
        colors = ['cornflowerblue', 'k', 'lightsage', 'b']
    elif nb_cost == 3:
        labels = ['Data Fid', 'Sparsity', 'Full']
        colors = ['cornflowerblue', 'k', 'b']

    plt.figure()
    for j in range(nb_cost):
        plt.plot(cost_list[:,j],label=labels[j], c=colors[j])

    plt.title('Cost Function')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.legend()
    plt.savefig(file_name+'.pdf')
    plt.close()

    # PSF only 
    if nb_cost == 4:
        plt.figure()
        plt.plot(cost_list[:,2],label='PSF update',c='lightsage')

        plt.title('Cost Function')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.legend()
        plt.savefig(file_name+'psf.pdf')
        plt.close()

    # cost, log scale
    plt.figure()
    for j in range(nb_cost):
        plt.plot(np.log10(cost_list[:,j]),label=labels[j], c=colors[j])

    plt.title('Cost Function')
    plt.xlabel('Iteration')
    plt.ylabel('$\log_{10}$ Cost')
    plt.legend()
    plt.savefig(file_name+'log.pdf')
    plt.close()
