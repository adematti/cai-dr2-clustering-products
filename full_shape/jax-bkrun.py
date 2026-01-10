"""
modified version of https://github.com/cosmodesi/cai-mock-benchmark/blob/main/dr2/compare_cutsky.py
salloc -N 1 -C gpu -t 02:00:00 --gpus 4 --qos interactive --account desi_g
salloc -N 1 -C "gpu&hbm80g" -t 02:00:00 --gpus 4 --qos interactive --account desi_g
salloc -N 1 -C "gpu&hbm80g" -t 00:10:00 --gpus 4 --qos interactive --account desi_g
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
srun -n 4 python jax-bkrun.py
"""

import os
import time
import logging
import itertools
from pathlib import Path

import numpy as np
from datetime import datetime
from time import time

from mockfactory import setup_logging
import lsstypes as types

from tools import select_region, combine_regions, get_catalog_dir, get_catalog_fn, get_power_fn, get_clustering_positions_weights, compute_fkp_effective_redshift

logger = logging.getLogger('bispectrum')

def compute_jaxpower_mesh3_spectrum(output_fn, get_data, get_randoms, get_shifted=None, cache=None, basis='scoccimarro', ells=[0, 2], los='local', buffer_size=0, **attrs):
    import jax
    from jaxpower import (ParticleField, FKPField, compute_fkp3_normalization, compute_fkp3_shotnoise, BinMesh3SpectrumPoles, get_mesh_attrs, compute_mesh3_spectrum)
    
    data, randoms = get_data(), get_randoms()
    mattrs = get_mesh_attrs(data[0], randoms[0], check=True, **attrs)
    data = list(data)
    
    bitwise_weights = None
    if len(data[1]) > 1:
        bitwise_weights = list(data[1])
        from cucount.jax import BitwiseWeight
        from cucount.numpy import reformat_bitarrays
        data[1] = individual_weight = bitwise_weights[0] * BitwiseWeight(weights=bitwise_weights[1:], p_correction_nbits=False)(bitwise_weights[1:])  # individual weight * IIP weight
    else:  # no bitwise_weights
        data[1] = individual_weight = data[1][0]
        
    data = ParticleField(*data, attrs=mattrs, exchange=True, backend='mpi')
    wsum_data1 = data.sum()
    randoms = ParticleField(randoms[0], randoms[1][0], attrs=mattrs, exchange=True, backend='mpi')
    fkp = FKPField(data, randoms)
    if cache is None: cache = {}
    bin = cache.get(f'bin_mesh3_spectrum_{basis}', None)
    #if bin is None: 
    if bin is None: bin = BinMesh3SpectrumPoles(mattrs, edges={'step': 0.01 if 'scoccimarro' in basis else 0.005}, basis=basis, ells=ells, buffer_size=buffer_size)
    cache.setdefault(f'bin_mesh3_spectrum_{basis}', bin)
    norm = compute_fkp3_normalization(fkp, split=42, bin=bin, cellsize=10)
    if get_shifted is not None:
        del fkp, randoms
        randoms = ParticleField(*get_shifted(), attrs=mattrs, exchange=True, backend='mpi')
        fkp = FKPField(data, randoms)
    kw = dict(resampler='tsc', interlacing=3, compensate=True)
    num_shotnoise = compute_fkp3_shotnoise(fkp, los=los, bin=bin, **kw)
    jitted_compute_mesh3_spectrum = jax.jit(compute_mesh3_spectrum, static_argnames=['los'], donate_argnums=[0])
    mesh = fkp.paint(**kw, out='real')
    spectrum = jitted_compute_mesh3_spectrum(mesh, los=los, bin=bin)
    mattrs = {name: mattrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']}
    spectrum = spectrum.clone(attrs=dict(los=los, wsum_data1=wsum_data1, **mattrs))
    if output_fn is not None and jax.process_index() == 0:
        logger.info(f'Writing to {output_fn}')
        spectrum.write(output_fn)
    return spectrum


def compute_jaxpower_window_mesh3_spectrum(output_fn, get_randoms, spectrum_fn=None, kind='smooth', **kwargs):
    from jax import numpy as jnp
    from jaxpower import (ParticleField, BinMesh3SpectrumPoles, BinMesh3CorrelationPoles, compute_mesh3_correlation, compute_fkp3_shotnoise, compute_smooth3_spectrum_window, MeshAttrs, get_smooth3_window_bin_attrs, interpolate_window_function, split_particles, read)
    spectrum = read(spectrum_fn)
    mattrs = MeshAttrs(**{name: spectrum.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']})
    #mattrs = mattrs.clone(meshsize=mattrs.meshsize // 4)
    los = spectrum.attrs['los']
    pole = next(iter(spectrum))
    ells, norm, edges, basis = spectrum.ells, pole.values('norm')[0], pole.edges('k'), pole.basis
    _, index = np.unique(pole.coords('k', center='mid')[..., 0], return_index=True)
    edges = edges[index, 0]
    edges = np.insert(edges[:, 1], 0, edges[0, 0])
    bin = BinMesh3SpectrumPoles(mattrs, **(dict(edges=edges, ells=ells, basis=basis) | kwargs), mask_edges='')
    #bin = BinMesh3SpectrumPoles(mattrs, **(dict(edges={'step': 0.005}, ells=ells, basis=basis) | kwargs), mask_edges='')
    step = bin.edges1d[0][-1, 1] - bin.edges1d[0][-1, 0]
    edgesin = np.arange(0., 1.5 * bin.edges1d[0].max(), step) # / 2.)
    edgesin = jnp.column_stack([edgesin[:-1], edgesin[1:]]) 
    output_fn = str(output_fn)
    
    randoms = ParticleField(*get_randoms(), attrs=mattrs, exchange=True, backend='mpi')
    zeff = compute_fkp_effective_redshift(randoms, order=3)

    kind = 'smooth'
    #kind = 'infinite'
    if kind == 'smooth':
        correlations = []
        kw, ellsin = get_smooth3_window_bin_attrs(ells, ellsin=2, fields=[1] * 3, return_ellsin=True)
        compute_mesh3_correlation = jax.jit(compute_mesh3_correlation, static_argnames=['los'], donate_argnums=[0, 1])

        coords = jnp.logspace(-3, 5, 4 * 1024)
        scales = [1, 4]
        b, c = mattrs.boxsize.min(), mattrs.cellsize.min()
        edges = [np.concatenate([np.arange(11) * c, np.arange(11 * c, 0.3 * b, 4 * c)]),
                np.concatenate([np.arange(11) * scales[1] * c, np.arange(11 * scales[1] * c, 2 * b, 4 * scales[1] * c)])]

        for scale, edges in zip(scales, edges):

            mattrs2 = mattrs.clone(boxsize=scale * mattrs.boxsize)
            kw_paint = dict(resampler='tsc', interlacing=3, compensate=True)
            sbin = BinMesh3CorrelationPoles(mattrs2, edges=edges, **kw, buffer_size=40)  # kcut=(0., mattrs2.knyq.min()))
            #num_shotnoise = compute_fkp3_shotnoise(randoms2, bin=sbin, **kw_paint)
            meshes = []
            for _ in split_particles(randoms.clone(attrs=mattrs2, exchange=True, backend='jax'), None, None, seed=42):
                alpha = spectrum.attrs['wsum_data1'] / _.sum()
                meshes.append(alpha * _.paint(**kw_paint, out='real'))
            correlation = compute_mesh3_correlation(*meshes, bin=sbin, los=los).clone(norm=[norm] * len(sbin.ells))
            if output_fn is not None and jax.process_index() == 0:
                correlation_fn = output_fn.replace('window_mesh3_spectrum', f'window_correlation{scale:d}_bessel_mesh3_spectrum')
                logger.info(f'Writing to {correlation_fn}')
                correlation.write(correlation_fn)
            correlation = interpolate_window_function(correlation.unravel(), coords=coords, order=3)
            correlations.append(correlation)

        coords = list(next(iter(correlations[0])).coords().values())
        limit = 0.25 * mattrs.boxsize.min()
        mask = (coords[0] < limit)[:, None] * (coords[1] < limit)[None, :]
        weights = [jnp.maximum(mask, 1e-6), jnp.maximum(~mask, 1e-6)]
        correlation = correlations[0].sum(correlations, weights=weights)
        flags = ('fftlog',)
        if output_fn is not None and jax.process_index() == 0:
            correlation_fn = output_fn.replace('window_mesh3_spectrum', 'window_correlation_bessel_mesh3_spectrum')
            logger.info(f'Writing to {correlation_fn}')
            correlation.write(correlation_fn)

        window = compute_smooth3_spectrum_window(correlation, edgesin=edgesin, ellsin=ellsin, bin=bin, flags=flags)
    else:
        raise NotImplementedError
    window = window.clone(observable=window.observable.map(lambda pole: pole.clone(norm=norm * np.ones_like(pole.values('norm')))))
    for pole in window.theory: pole._meta['z'] = zeff
    if output_fn is not None and jax.process_index() == 0:
        #output_fn = output_fn.replace('sugiyama-diagonal', 'sugiyama-diagonal_damped')
        logger.info(f'Writing to {output_fn}')
        window.write(output_fn)
    return window

def collect_argparser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--todo',  help='what do you want to compute?', type=str, nargs='+',choices=['mesh3_spectrum_scoccimarro', 'mesh3_spectrum_sugiyama', 'window_mesh3_spectrum_sugiyama','combine'], default=['mesh3_spectrum_sugiyama'])
    parser.add_argument('--tracer',    help='tracer(s) to be selected - 2 for cross-correlation', type=str, nargs='+', default=['QSO'])
    parser.add_argument('--zrange', nargs='+', type=str, default=None, help='Redshift bins')
    parser.add_argument('--basedir', help='where to find catalogs', type=str, default='/dvs_ro/cfs/cdirs/desi/survey/catalogs/')
    parser.add_argument('--outdir',  help="base directory for output, default is SCRATCH", type=str, default=os.getenv('PSCRATCH'))
    parser.add_argument('--survey',  help='e.g., SV3 or main', type=str, choices=['SV3', 'DA02', 'main', 'Y1','DA2'], default='Y1')
    parser.add_argument('--verspec', help='version for redshifts', type=str, default='iron')
    parser.add_argument('--version', help='catalog version', type=str, default='test')
    parser.add_argument('--regions', help='regions', type=str, nargs='*', choices=['N', 'S', 'NGC', 'SGC', 'NGCnoN', 'SGCnoDES'], default=None)
    
    parser.add_argument('--weight_type',  help='types of weights to use for tracer1; "default" just uses WEIGHT column', type=str, default='default_FKP')

    parser.add_argument('--boxsize',  help='box size', type=float, default=10000.)
    parser.add_argument('--cellsize', help='cell size', default=10)
    parser.add_argument('--nran', help='number of random files to combine together (1-18 available)', type=int, default=10)
    parser.add_argument('--P0',  help='value of P0 to use in FKP weights (None defaults to WEIGHT_FKP)', type=float, default=None)
    parser.add_argument('--P02', help='value of P0 to use in FKP weights (None defaults to WEIGHT_FKP) of tacer2', type=float, default=None)
    parser.add_argument('--recon_dir', help='if recon catalogs are in a subdirectory, put that here', type=str, default='n')
    parser.add_argument('--ric_dir', help='where to find ric/noric randoms', type=str, default=None)
    
    return parser.parse_args()
    
if __name__ == '__main__':
    import jax
    from jax import config
    config.update('jax_enable_x64', True)
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'
    jax.distributed.initialize()
    from jaxpower.mesh import create_sharding_mesh
    
    # gather arguments
    args = collect_argparser()

    #
    setup_logging()
    logger.info(f"Arguments: {args}")

    # define important variables given by input arguments
    zrange = [float(iz) for iz in args.zrange]
    if len(zrange)%2 != 0:
        raise ValueError(f"Redshift ranges should be given in pairs, got {zrange}")
    else:
        zrange = list(zip(zrange[::2],zrange[1::2]))   
    survey   = args.survey
    verspec  = args.verspec
    version  = args.version
    regions  = args.regions
    boxsize  = args.boxsize
    cellsize = args.cellsize
    nran =  args.nran
    input_fmt = 'h5'

    # We allow for cross-correlation 
    tracer = args.tracer[0]
    if len(args.tracer) > 1:
        raise ValueError('Provide <= 1 tracers!')

    # We allow for differenct weight_types for both tracers when computing cross-correlations
    weight_type  = args.weight_type

    # get input directory (location of data and random catalogs)
    if os.path.normpath(args.basedir) == os.path.normpath('/dvs_ro/cfs/cdirs/desi/survey/catalogs/'):
        catalog_dir = get_catalog_dir(base_dir=args.basedir, survey=survey, verspec=verspec, version=version)
    else:
        catalog_dir = args.basedir
    # get ouput directory (save to scratch by default)
    out_dir = args.outdir

    # what to do?
    todo = args.todo

    # file to log date and time a file was made 
    # and how much time it took to generate it.
    logfn = out_dir+'clustering_log.txt'
    if jax.process_index() == 0:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if not os.path.isfile(logfn):
            with open(logfn, "a") as logf:
                logf.write("File to log date and time a file was made and how much time it took to generate it.\n")
    
    # iterate over redshift bins then regions
    for i_zrange in zrange:
        for region in regions:
            cache = {}
            # Collect common arguments and then tracer specific variables in dictionaries
            common_args = dict(zrange=i_zrange, nran=nran, base_dir=catalog_dir, region=region)
            tracer_args  = dict(tracer=tracer, weight_type=weight_type)
    
            # Collect spectrum arguments in a dictonary
            spectrum_args = dict(boxsize=boxsize, cellsize=cellsize, ells=(0, 2, 4))
            
            # Collect other optional arguments in a dictionary and remove them from the 'weight_type' keys
            meas_args = dict()
            if weight_type.endswith('_thetacut'):
                tracer_args['weight_type']  = weight_type[:-len('_thetacut')]
                meas_args['cut'] = 'theta'
            with_auw = weight_type.endswith('_auw')
            if with_auw:
                tracer_args['weight_type'] = weight_type[:-len('_auw')]
            
            # get the filenames for the data and randoms
            if region in ['SGC','NGC']:
                data_kind, randoms_kind = 'data','randoms'
            else:
                # Full clustering catalogs do not have FKP weights
                # So I made it so kind='full_data_clus' returns both NGC and SGC files.
                # Then the splitting is handled by get_clustering_positions_weights.
                data_kind, randoms_kind = 'full_data_clus','full_randoms_clus'
            data_fn = get_catalog_fn(kind=data_kind, fmt=input_fmt, **common_args, **tracer_args)
            all_randoms_fn = get_catalog_fn(kind=randoms_kind, fmt=input_fmt, **common_args, **tracer_args)
            
            # Load data and randoms
            get_data = lambda: get_clustering_positions_weights(data_fn, kind='data', **common_args, **tracer_args)
            get_randoms = lambda: get_clustering_positions_weights(*all_randoms_fn, kind='randoms', **common_args, **tracer_args)
            
            # if using recon (not implemented yet)
            get_shifted   = None
           
            spectrum_args |= meas_args
            
            # Collect ouput fn arguments 
            output_fn_args = dict(base_dir=out_dir, file_type='h5', region=region, tracer=tracer,
                                  zmin=i_zrange[0], zmax=i_zrange[1], weight_type=weight_type,
                                  nran=nran, P0=None, P02=None, ric_dir=None) | meas_args
    
            # Compute bispectrum with jax-power
            if 'mesh3_spectrum_scoccimarro' in todo:
                bispectrum_args = spectrum_args | dict(basis='scoccimarro', ells=[0, 2])
                output_fn = get_power_fn(**output_fn_args, boxsize=boxsize, cellsize=cellsize, kind=f'mesh3_spectrum_poles_{bispectrum_args["basis"]}')
                with create_sharding_mesh() as sharding_mesh:
                    compute_jaxpower_mesh3_spectrum(output_fn, get_data, get_randoms, get_shifted=get_shifted, cache=cache, **bispectrum_args)
                    jax.clear_caches() 
    
            if 'mesh3_spectrum_sugiyama' in todo:
                t0 = time() # time start
                bispectrum_args = spectrum_args | dict(basis='sugiyama-diagonal', ells=[(0, 0, 0), (2, 0, 2)])
                output_fn = get_power_fn(**output_fn_args, boxsize=boxsize, cellsize=cellsize, kind=f'mesh3_spectrum_poles_{bispectrum_args["basis"]}')
                with create_sharding_mesh() as sharding_mesh:
                    compute_jaxpower_mesh3_spectrum(output_fn, get_data, get_randoms, get_shifted=get_shifted, cache=cache, **bispectrum_args)
                if jax.process_index() == 0:
                    with open(logfn, "a") as logf:
                        logf.write(f"{str(datetime.now())} - {str(output_fn).split('/')[-1]} generated in {time()-t0:.2f} seconds\n")
    
            if 'window_mesh3_spectrum_sugiyama' in todo and iimock == 1:
                jax.experimental.multihost_utils.sync_global_devices("spectrum")
                spectrum_fn = get_measurement_fn(imock=imock, **catalog_args, kind=f'mesh3_spectrum_poles_sugiyama-diagonal')
                output_fn = get_power_fn(**output_fn_args, boxsize=boxsize, cellsize=cellsize, kind=f'window_mesh3_spectrum_poles_sugiyama-diagonal')
                with create_sharding_mesh() as sharding_mesh:
                    compute_jaxpower_window_mesh3_spectrum(output_fn, get_randoms, spectrum_fn=spectrum_fn)
                    jax.clear_caches()
                    
        if 'combine' in todo:
            jax.experimental.multihost_utils.sync_global_devices("spectrum")
            if jax.process_index() == 0:
                t0 = time()
                for kind in [f'mesh3_spectrum_poles_{bispectrum_args["basis"]}']:
                    kw = dict(kind=kind, **output_fn_args, boxsize=boxsize, cellsize=cellsize)
                    fns = [get_power_fn(**(kw | dict(region=region))) for region in regions]
                    if ('NGC' in regions) and ('SGC' in regions):
                        region_comb = 'GCcomb' 
                    elif ('N' in regions) and ('S' in regions):
                        region_comb = 'NS'
                    elif ('NGCnoN' in regions) and ('SGC' in regions):
                        region_comb = 'GCcomb_noNorth'
                    elif ('NGC' in regions) and ('SGCnoDES' in regions):
                        region_comb = 'GCcomb_noDES'
                    else: 
                        raise ValueError(f'Combining regions is not implemented for {regions}')
                    output_fn = get_power_fn(**(kw | dict(region=region_comb)))
                    print(fns,output_fn)
                    combine_regions(output_fn, fns, logger=logger)
                    with open(logfn, "a") as logf:
                        logf.write(f"{str(datetime.now())} - {str(output_fn).split('/')[-1]} generated in {time()-t0:.2f} seconds from combining {str(fns[0]).split('/')[-1]} and {str(fns[1]).split('/')[-1]}\n")
                
    jax.distributed.shutdown()