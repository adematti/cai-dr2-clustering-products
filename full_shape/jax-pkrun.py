"""
modified version of https://github.com/cosmodesi/cai-mock-benchmark/blob/main/dr2/data_pip.py
salloc -N 1 -C gpu -t 02:00:00 --gpus 4 --qos interactive --account desi_g
salloc -N 1 -C gpu -t 00:10:00 --gpus 4 --qos interactive --account desi_g
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh test
srun -n 4 python pkrun.py
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

logger = logging.getLogger('pkrun') 

def compute_jaxpower_mesh2_spectrum(output_fn, get_data, get_randoms, get_data_2=None, get_randoms_2=None, 
                                    get_shifted=None, cache=None, ells=(0, 2, 4), los='firstpoint', **attrs):
    import jax
    from jaxpower import (ParticleField, FKPField, compute_fkp2_normalization, compute_fkp2_shotnoise, BinMesh2SpectrumPoles, get_mesh_attrs, compute_mesh2_spectrum)

    t0 = time()
    data, randoms = get_data(), get_randoms()
    logger.info(f'Loaded data and randoms in: {time() - t0:.2f}.')
    #data,randoms = get_data, get_randoms
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
    randoms = ParticleField(randoms[0], randoms[1][0], attrs=mattrs, exchange=True, backend='mpi')
    fkp = FKPField(data, randoms)
    if cache is None: cache = {}
    bin = cache.get('bin_mesh2_spectrum', None)
    if bin is None: bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.001}, ells=ells)
    cache.setdefault('bin_mesh2_spectrum', bin)
    norm = compute_fkp2_normalization(fkp, bin=bin, cellsize=10)
    if get_shifted is not None:
        del fkp, randoms
        randoms = ParticleField(*get_shifted(), attrs=mattrs, exchange=True, backend='mpi')
        fkp = FKPField(data, randoms)
    num_shotnoise = compute_fkp2_shotnoise(fkp, bin=bin)
    mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    #t0 = time.time()
    wsum_data1 = data.sum()
    del fkp, data, randoms
    jitted_compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'], donate_argnums=[0])
    #jitted_compute_mesh2_spectrum = compute_mesh2_spectrum
    spectrum = jitted_compute_mesh2_spectrum(mesh, bin=bin, los=los).clone(norm=norm, num_shotnoise=num_shotnoise)
    mattrs = {name: mattrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']}
    spectrum = spectrum.clone(attrs=dict(los=los, wsum_data1=wsum_data1, **mattrs))
    if output_fn is not None and jax.process_index() == 0:
        logger.info(f'Writing to {output_fn}')
        spectrum.write(output_fn)
    jax.block_until_ready(spectrum)
    #logger.info(f'Elapsed time: {time.time() - t0:.2f}.')
    #del spectrum, bin
    return spectrum


def compute_jaxpower_window_mesh2_spectrum(output_fn, get_randoms, get_data=None, spectrum_fn=None, kind='smooth', **kwargs):
    from jax import numpy as jnp
    from jaxpower import (ParticleField, compute_mesh2_spectrum_window, BinMesh2SpectrumPoles, BinMesh2CorrelationPoles, compute_mesh2_correlation, compute_fkp2_shotnoise, compute_smooth2_spectrum_window, MeshAttrs, get_smooth2_window_bin_attrs, interpolate_window_function, compute_mesh2_spectrum, split_particles, read)
    spectrum = read(spectrum_fn)
    mattrs = MeshAttrs(**{name: spectrum.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']})
    los = spectrum.attrs['los']
    pole = next(iter(spectrum))
    ells, norm, edges = spectrum.ells, pole.values('norm')[0], pole.edges('k')
    bin = BinMesh2SpectrumPoles(mattrs, **(dict(edges=edges, ells=ells) | kwargs))
    step = bin.edges[-1, 1] - bin.edges[-1, 0]
    edgesin = np.arange(0., 1.2 * bin.edges.max(), step)
    edgesin = jnp.column_stack([edgesin[:-1], edgesin[1:]]) 
    ellsin = [0, 2, 4]
    output_fn = str(output_fn)

    #randoms = ParticleField(*get_randoms(), attrs=mattrs, exchange=True, backend='mpi')
    randoms = get_randoms()
    randoms = ParticleField(randoms[0], randoms[1][0], attrs=mattrs, exchange=True, backend='mpi')
    zeff = compute_fkp_effective_redshift(randoms, order=2)
    #if get_data is not None:
    #    from jaxpower import FKPField
    #    data = ParticleField(*get_data(), attrs=mattrs, exchange=True, backend='mpi')
    #    zeff = compute_fkp_effective_redshift(FKPField(data=data, randoms=randoms))
    randoms = spectrum.attrs['wsum_data1'] / randoms.sum() * randoms

    kind = 'smooth'
    #kind = 'infinite'

    if kind == 'smooth':
        correlations = []
        kw = get_smooth2_window_bin_attrs(ells, ellsin)
        compute_mesh2_correlation = jax.jit(compute_mesh2_correlation, static_argnames=['los'], donate_argnums=[0, 1])
        # Window computed in configuration space, summing Bessel over the Fourier-space mesh
        coords = jnp.logspace(-3, 5, 4 * 1024)
        for scale in [1, 4]:
            mattrs2 = mattrs.clone(boxsize=scale * mattrs.boxsize) #, meshsize=800)
            kw_paint = dict(resampler='tsc', interlacing=3, compensate=True)
            meshes = []
            for _ in split_particles(randoms.clone(attrs=mattrs2, exchange=True, backend='jax'), None, seed=42):
                alpha = spectrum.attrs['wsum_data1'] / _.sum()
                meshes.append(alpha * _.paint(**kw_paint, out='real'))
            sbin = BinMesh2CorrelationPoles(mattrs2, edges=np.arange(0., mattrs2.boxsize.min() / 2., mattrs2.cellsize.min()), **kw, basis='bessel') 
            correlation = compute_mesh2_correlation(*meshes, bin=sbin, los=los).clone(norm=[norm] * len(sbin.ells))
            del meshes
            correlation_fn = output_fn.replace('window_mesh2_spectrum', f'window_correlation{scale:d}_bessel_mesh2_spectrum')
            if jax.process_index() == 0:
                logger.info(f'Writing to {correlation_fn}')
                correlation.write(correlation_fn)
            correlation = interpolate_window_function(correlation, coords=coords, order=3)
            correlations.append(correlation)
        limits = [0, 0.4 * mattrs.boxsize.min(), 2. * mattrs.boxsize.max()]
        weights = [jnp.maximum((coords >= limits[i]) & (coords < limits[i + 1]), 1e-10) for i in range(len(limits) - 1)]
        correlation = correlations[0].sum(correlations, weights=weights)
        flags = ('fftlog',)
        if output_fn is not None and jax.process_index() == 0:
            correlation_fn = output_fn.replace('window_mesh2_spectrum', 'window_correlation_bessel_mesh2_spectrum')
            logger.info(f'Writing to {correlation_fn}')
            correlation.write(correlation_fn)
        window = compute_smooth2_spectrum_window(correlation, edgesin=edgesin, ellsin=ellsin, bin=bin, flags=flags)
    else:
        kw_paint = dict(resampler='tsc', interlacing=3, compensate=True)
        meshes = []
        for _ in split_particles(randoms, None, seed=42):
            alpha = spectrum.attrs['wsum_data1'] / _.sum()
            meshes.append(alpha * _.paint(**kw_paint, out='real'))
        window = compute_mesh2_spectrum_window(*meshes, edgesin=edgesin, ellsin=ellsin, los=los, bin=bin, pbar=True, flags=('infinite',), norm=norm)
    window = window.clone(observable=window.observable.map(lambda pole: pole.clone(norm=norm * np.ones_like(pole.values('norm')))))
    for pole in window.theory: pole._meta['z'] = zeff
    if output_fn is not None and jax.process_index() == 0:
        logger.info(f'Writing to {output_fn}')
        window.write(output_fn)
    return window

    
def collect_argparser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--todo',  help='what do you want to compute?', type=str, nargs='+',choices=['mesh2_spectrum','window_mesh2_spectrum','combine'], default=['mesh2_spectrum'])
    parser.add_argument('--tracer',    help='tracer(s) to be selected - 2 for cross-correlation', type=str, nargs='+', default=['QSO'])
    parser.add_argument('--zrange', nargs='+', type=str, default=None, help='Redshift bins')
    parser.add_argument('--basedir', help='where to find catalogs', type=str, default='/dvs_ro/cfs/cdirs/desi/survey/catalogs/')
    parser.add_argument('--outdir',  help="base directory for output, default is SCRATCH", type=str, default=os.getenv('PSCRATCH'))
    parser.add_argument('--survey',  help='e.g., SV3 or main', type=str, choices=['SV3', 'DA02', 'main', 'Y1','DA2'], default='Y1')
    parser.add_argument('--verspec', help='version for redshifts', type=str, default='iron')
    parser.add_argument('--version', help='catalog version', type=str, default='test')
    parser.add_argument('--regions', help='regions', type=str, nargs='*', choices=['N', 'S', 'NGC', 'SGC', 'NGCnoN', 'SGCnoDES'], default=None)
    
    parser.add_argument('--weight_type',  help='types of weights to use for tracer1; "default" just uses WEIGHT column', type=str, default='default_FKP')
    parser.add_argument('--weight_type2', help='types of weights to use for tracer2; by default uses same as tracer1 (weight_type)', type=str, default=None)

    parser.add_argument('--boxsize',  help='box size', type=float, default=10000.)
    # parser.add_argument('--boxsize', help='box size, can be multiple input e.g. [8000,8000,8000]', type=float, nargs='*', default=8000.)
    # parser.add_argument('--nmesh', help='mesh size', default=None)
    parser.add_argument('--cellsize', help='cell size', default=10.)
    parser.add_argument('--nran', help='number of random files to combine together (1-18 available)', type=int, default=10)
    # parser.add_argument('--calc_win', help='also calculate window?; use "y" for yes', default='n')
    # parser.add_argument('--rebinning', help='whether to rebin the pk or just keep the original .npy file', default='n')
    # parser.add_argument('--thetacut', help='apply this theta-cut, standard 0.05', type=float, default=None)
    parser.add_argument('--P0',  help='value of P0 to use in FKP weights (None defaults to WEIGHT_FKP)', type=float, default=None)
    parser.add_argument('--P02', help='value of P0 to use in FKP weights (None defaults to WEIGHT_FKP) of tacer2', type=float, default=None)
    parser.add_argument('--recon_dir', help='if recon catalogs are in a subdirectory, put that here', type=str, default='n')
    # only relevant for reconstruction
    # parser.add_argument('--rec_type', help='reconstruction algorithm + reconstruction convention', choices=['IFTrecsym', 'IFTreciso', 'MGrecsym', 'MGreciso'], type=str, default=None)
    # parser.add_argument('--use_rands_for_tracer1', help='replace tracer 1 data for randoms?; use "y" for yes', default='n')
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
    # if mpicomm.rank == 0:
    #     logger.info(f"Arguments: {args}")
    logger.info(f"Arguments: {args}")

    # define important variables given by input arguments
    zrange = [float(iz) for iz in args.zrange]
    if len(zrange)%2 !=0:
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

    # We allow for cross-correlation 
    tracer, tracer2 = args.tracer[0], None
    if len(args.tracer) > 1:
        tracer2 = args.tracer[1]
        if len(args.tracer) > 2:
            raise ValueError('Provide <= 2 tracers!')
    if tracer2 == tracer:
        tracer2 = None # otherwise counting of self-pairs

    # We allow for differenct weight_types for both tracers when computing cross-correlations
    weight_type  = args.weight_type
    weight_type2 = args.weight_type if args.weight_type2 is None else args.weight_type2  

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
            # if 'BGS' in tracer:
            #     tracer = 'BGS_BRIGHT-21.5' if 'dr1' in version else 'BGS_BRIGHT-21.35'
            # Collect common arguments and then tracer specific variables in dictionaries
            common_args = dict(zrange=i_zrange, nran=nran, base_dir=catalog_dir, region=region)
            tracer_args  = dict(tracer=tracer, weight_type=weight_type)
            if tracer2 is not None:
                tracer2_args = dict(tracer=tracer2, weight_type=weight_type2)
    
            # Collect spectrum arguments in a dictonary
            spectrum_args = dict(boxsize=boxsize, cellsize=cellsize, ells=(0, 2, 4))
            
            # Collect other optional arguments in a dictionary and remove them from the 'weight_type' keys
            meas_args = dict()
            if weight_type.endswith('_thetacut'):
                tracer_args['weight_type']  = weight_type[:-len('_thetacut')]
                if tracer2 is not None:
                    tracer2_args['weight_type'] = weight_type2[:-len('_thetacut')]
                meas_args['cut'] = 'theta'
            with_auw = weight_type.endswith('_auw')
            if with_auw:
                tracer_args['weight_type'] = weight_type[:-len('_auw')]
                if tracer2 is not None:
                    tracer2_args['weight_type'] = weight_type2[:-len('_auw')]
            
            # get the filenames for the data and randoms
            if region in ['SGC','NGC']:
                data_kind, randoms_kind = 'data','randoms'
            else:
                # Not very useful, full clustering catalogs do not have FKP weights
                # So I made it so kind='full_data_clus' returns both NGC and SGC files.
                # Then the splitting is handled by get_clustering_positions_weights.
                data_kind, randoms_kind = 'full_data_clus','full_randoms_clus'
            data_fn = get_catalog_fn(kind=data_kind, **common_args, **tracer_args)
            all_randoms_fn = get_catalog_fn(kind=randoms_kind, **common_args, **tracer_args)
            if tracer2 is not None:
                data_fn_2 = get_catalog_fn(kind=data_kind, **common_args, **tracer2_args)
                all_randoms_fn_2 = get_catalog_fn(kind=randoms_kind, **common_args, **tracer2_args)
            # bitwise option not implemented yet
            # if 'bitwise' in tracer['weight_type']:
            #     catalog_args['ntmp'] = compute_ntmp(get_catalog_fn(kind='full_data', **tracer_args))
    
            # Load data and randoms
            get_data = lambda: get_clustering_positions_weights(data_fn, kind='data', **common_args, **tracer_args)
            get_randoms = lambda: get_clustering_positions_weights(*all_randoms_fn, kind='randoms', **common_args, **tracer_args)
            if tracer2 is not None:
                get_data_2 = lambda: get_clustering_positions_weights(data_fn_2, kind='data', **common_args, **tracer2_args)
                get_randoms_2 = lambda: get_clustering_positions_weights(*all_randoms_fn_2, kind='randoms', **common_args, **tracer2_args)
            else: 
                get_data_2 = None
                get_randoms_2 = None
            # if using recon (not implemented yet)
            get_shifted   = None
            get_shifted_2 = None
            
            # Angular upweighting not implemented yet
            # if 'auw' in todo:
            #     full_data_fn = get_catalog_fn(kind='full_data', **catalog_args)
            #     all_full_randoms_fn = get_catalog_fn(kind='full_randoms', **catalog_args)
            #     get_full_data = lambda kind: get_full_rdw(full_data_fn, kind=kind, **catalog_args)
            #     get_full_randoms = lambda kind: get_full_rdw(*all_full_randoms_fn, kind=kind, **catalog_args)
        
            #     output_fn = get_measurement_fn(**catalog_args, kind='angular_upweights')
            #     with create_sharding_mesh() as sharding_mesh:
            #         compute_angular_upweights(output_fn, get_full_data, get_full_randoms)
        
            # if with_auw:
            #     jax.experimental.multihost_utils.sync_global_devices('auw')
            #     meas_args['auw'] = types.read(get_measurement_fn(**catalog_args, kind='angular_upweights'))
            
            spectrum_args |= meas_args
            
            # Collect ouput fn arguments 
            output_fn_args = dict(base_dir=out_dir, file_type='h5', region=region, tracer=tracer, tracer2=tracer2, 
                                  zmin=i_zrange[0], zmax=i_zrange[1], weight_type=weight_type, weight_type2=weight_type2, 
                                  nran=nran, P0=None, P02=None, ric_dir=None) | meas_args
            
            # Compute power spectrum with jax-power
            if 'mesh2_spectrum' in todo:
                t0 = time() # time start
                output_fn = get_power_fn(**output_fn_args, boxsize=boxsize, cellsize=cellsize, kind='mesh2_spectrum_poles')
                spectrum_args2 = dict(spectrum_args)
                with create_sharding_mesh() as sharding_mesh:
                    compute_jaxpower_mesh2_spectrum(output_fn, get_data, get_randoms, get_data_2=get_data_2, get_randoms_2=get_randoms_2,
                                                    get_shifted=get_shifted, cache=cache, **spectrum_args2)
                    jax.clear_caches()
                if jax.process_index() == 0:
                    with open(logfn, "a") as logf:
                        logf.write(f"{str(datetime.now())} - {str(output_fn).split('/')[-1]} generated in {time()-t0:.2f} seconds\n")
            
            # Compute window matrix with jax-power
            if 'window_mesh2_spectrum' in todo:
                jax.experimental.multihost_utils.sync_global_devices("spectrum")
                spectrum_fn = get_power_fn(**output_fn_args, boxsize=boxsize, cellsize=cellsize, kind='mesh2_spectrum_poles')
                output_fn   = get_power_fn(**output_fn_args, boxsize=boxsize, cellsize=cellsize, kind='window_mesh2_spectrum_poles')
                with create_sharding_mesh() as sharding_mesh:
                    compute_jaxpower_window_mesh2_spectrum(output_fn, get_randoms, get_data=get_data, spectrum_fn=spectrum_fn)
                    jax.clear_caches()
                        
        
        if 'combine' in todo:
            jax.experimental.multihost_utils.sync_global_devices("spectrum")
            if jax.process_index() == 0:
                t0 = time()
                for kind in ['mesh2_spectrum_poles']:
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
