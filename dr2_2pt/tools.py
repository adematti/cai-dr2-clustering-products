import os 
import numpy as np
from pathlib import Path

from mockfactory import Catalog, setup_logging, sky_to_cartesian
import lsstypes as types
from astropy.table import Table

def load_footprint():
    #global footprint
    from regressis import footprint
    footprint = footprint.DR9Footprint(256, mask_lmc=False, clear_south=True, mask_around_des=False, cut_desi=False)
    return footprint
    
def select_region(ra, dec, region=None):
    import healpy as hp
    # print('select', region)
    if region in [None, 'ALL', 'GCcomb']:
        return np.ones_like(ra, dtype='?')
    mask_ngc = (ra > 100 - dec)
    mask_ngc &= (ra < 280 + dec)
    mask_n = mask_ngc & (dec > 32.375)
    mask_s = (~mask_n) & (dec > -25.)
    if region == 'NGC':
        return mask_ngc
    if region == 'SGC':
        return ~mask_ngc
    if region == 'N':
        return mask_n
    if region == 'S':
        return mask_s
    if region == 'SNGC':
        return mask_ngc & mask_s
    if region == 'SSGC':
        return (~mask_ngc) & mask_s
    if region == 'NGCnoN':
        return mask_ngc & (~mask_n)
    # if region == 'GCcomb_noNorth':
    #     return ~mask_n
    north, south, des = load_footprint().get_imaging_surveys()
    mask_des = des[hp.ang2pix(256, ra, dec, nest=True, lonlat=True)]
    if region == 'DES':
        return mask_des
    if region == 'SnoDES':
        return mask_s & (~mask_des)
    if region == 'SSGCnoDES':
        return (~mask_ngc) & mask_s & (~mask_des)
    if region == 'SGCnoDES':
        return (~mask_ngc) & (~mask_des)
    # if region == 'GCcomb_noDES':
    #     return ~mask_des
    raise ValueError('unknown region {}'.format(region))


def get_proposal_mattrs(tracer):
    if 'BGS' in tracer:
        mattrs = dict(boxsize=4000., cellsize=10)
    elif 'LRG+ELG' in tracer:
        mattrs = dict(boxsize=9000., cellsize=10)
    elif 'LRG' in tracer:
        mattrs = dict(boxsize=7000., cellsize=10)
    elif 'ELG' in tracer:
        mattrs = dict(boxsize=9000., cellsize=10)
    elif 'QSO' in tracer:
        mattrs = dict(boxsize=10000., cellsize=10)
    else:
        raise NotImplementedError(f'tracer {tracer} is unknown')
    #mattrs.update(cellsize=30)
    return mattrs


def get_catalog_dir(survey='Y1', verspec='iron', version='v1.2', base_dir='/global/cfs/cdirs/desi/survey/catalogs'):
    base_dir = Path(base_dir)
    return base_dir / survey / 'LSS' / verspec / 'LSScats' / version


def get_catalog_fn(base_dir='/global/cfs/cdirs/desi/survey/catalogs', kind='data', tracer='LRG', weight_type='bitwise', zrange=(0.8, 1.1), region='NGC', nran=10, fmt='h5', **kwargs):
    # if 'bitwise' in weight_type: is not implemented yet 
    data_dir = Path(base_dir)
    if kind == 'data':
        return data_dir / f'{tracer}_{region}_clustering.dat.{fmt}'
    if kind == 'randoms':
        return [data_dir / f'{tracer}_{region}_{iran:d}_clustering.ran.{fmt}' for iran in range(nran)]
    if kind == 'full_data_clus':
        return [data_dir / f'{tracer}_{region}_clustering.dat.{fmt}' for region in ['NGC','SGC']]
    if kind == 'full_randoms_clus':
        return [data_dir / f'{tracer}_{region}_{iran:d}_clustering.ran.{fmt}' for iran in range(nran) for region in ['NGC','SGC']]
    if kind == 'full_data':
        return data_dir / f'{tracer}_full_HPmapcut.dat.{fmt}'
    if kind == 'full_randoms':
        return [data_dir / f'{tracer}_{iran:d}_full_HPmapcut.ran.{fmt}' for iran in range(nran)]


def get_power_fn(base_dir=os.getenv('PSCRATCH'), kind='', file_type='h5', region='', tracer='ELG', tracer2=None, zmin=0, zmax=np.inf, weight_type='default',
                 weight_type2='default', nran=None, option=None, cut=None, auw=None,
                 P0=None,P02=None,ric_dir=None, boxsize=None, cellsize=None):
    base_dir = Path(base_dir)
    weight_type1=weight_type
    if tracer2: tracer += '_' + tracer2
    if weight_type2!=weight_type: 
        weight_type += '_' + weight_type2
    # if rec_type: tracer += '_' + rec_type
    if region: tracer += '_' + region
    # if recon_dir != 'n':
    #     out_dir = out_dir[:-2] + recon_dir+'/pk/'
        
    if option:
        zmax = str(zmax) + option

    root = '{}_z{}-{}_{}'.format(tracer, zmin, zmax, weight_type)
    # if nran is not None:
    #     root += '_nran{}'.format(nran)
    # if cellsize is not None:
    #     root += '_cellsize{}'.format(cellsize)   
    # if isinstance(boxsize, list): 
    #     root += '_boxsize{}_{}_{}'.format(int(boxsize[0]),int(boxsize[1]),int(boxsize[2]))
    # else:
    #     root += '_boxsize{}'.format(int(boxsize))
    if P0 is not None:
        root += '_P0-{}'.format(P0)
    if P02 is not None:
        root += '_P02-{}'.format(P02)
    if auw is not None:
        root += '_auw'
    if cut is not None:
        root += '_thetacut'
    if ric_dir is not None:
        ric = 'noric' if 'noric' in ric_dir else 'ric'
        root += '_{}'.format(ric)
    return base_dir / '{}_{}.{}'.format(kind, root, file_type)
    
# def get_catalog_fn(version='dr1-v1.5', kind='data', tracer='LRG', weight_type='bitwise', zrange=(0.8, 1.1), region='NGC', nran=10, **kwargs):
#     desi_dir = Path('/dvs_ro/cfs/cdirs/desi/survey/catalogs/')
#     nran_full = 1
#     if version == 'dr1-v1.5':
#         base_dir = desi_dir / f'Y1/LSS/iron/LSScats'
#         if 'bitwise' in weight_type:
#             data_dir = base_dir / 'v1.5pip'
#         else:
#             data_dir = base_dir / 'v1.5'
#         if kind == 'data':
#             return data_dir / f'{tracer}_{region}_clustering.dat.fits'
#         if kind == 'randoms':
#             return [data_dir / f'{tracer}_{region}_{iran:d}_clustering.ran.fits' for iran in range(nran)]
#         if kind == 'full_data':
#             return data_dir / f'{tracer}_full_HPmapcut.dat.fits'
#         if kind == 'full_randoms':
#             return [data_dir / f'{tracer}_{iran:d}_full_HPmapcut.ran.fits' for iran in range(nran_full)]
#     elif version == 'dr2-v2':
#         base_dir = desi_dir / f'DA2/LSS/loa-v1/LSScats/v2'
#         if 'bitwise' in weight_type:
#             data_dir = base_dir / 'PIP'
#         else:
#             data_dir = base_dir / 'nonKP'
#         if kind == 'data':
#             return data_dir / f'{tracer}_{region}_clustering.dat.fits'
#         if kind == 'randoms':
#             return [data_dir / f'{tracer}_{region}_{iran:d}_clustering.ran.fits' for iran in range(nran)]
#         if kind == 'full_data':
#             return base_dir / f'{tracer}_full_HPmapcut.dat.fits'
#         if kind == 'full_randoms':
#             return [base_dir / f'{tracer}_{iran:d}_full_HPmapcut.ran.fits' for iran in range(nran_full)]
#     raise ValueError('issue with input args')


# def get_measurement_fn(kind='mesh2_spectrum_poles', version='dr2-v2', recon=None, tracer='LRG', region='NGC', zrange=(0.8, 1.1), cut=None, auw=None, weight_type='default', **kwargs):
#     #base_dir = Path(f'/global/cfs/projectdirs/desi/mocks/cai/mock-challenge-cutsky-dr2/')
#     base_dir = Path(f'/pscratch/sd/a/arosado/checks/power-spectrum/test/')
#     base_dir = base_dir / (f'unblinded_data_{recon}' if recon else 'unblinded_data')
#     if cut: cut = '_thetacut'
#     else: cut = ''
#     if auw: auw = '_auw'
#     else: auw = ''
#     return str(base_dir / f'{version}/{kind}_{tracer}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{region}_{weight_type}{auw}{cut}.h5')


def apply_wntmp(ntile, ntmp_table, method='ntmp'):
    frac_missing_pw, frac_zero_prob = ntmp_table
    if method == 'ntmp':
        toret = 1 - frac_missing_pw[ntile]
    elif method == 'ntzp':
        toret = 1 - frac_zero_prob[ntile]
    else:
        raise NotImplementedError(f'unknown method {method}')
    #ref = apply_wntmp_bak(ntile, frac_missing_pw, frac_zero_prob, ntile_range=[0,15], randoms=True)[0]
    #assert np.allclose(toret, ref)
    return toret

# Create a lookup table for set bits per byte
_popcount_lookuptable = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)

def popcount(*arrays):
    """
    Return number of 1 bits in each value of input array.
    Inspired from https://github.com/numpy/numpy/issues/16325.
    """
    # if not np.issubdtype(array.dtype, np.unsignedinteger):
    #     raise ValueError('input array must be an unsigned int dtype')
    toret = _popcount_lookuptable[arrays[0].view((np.uint8, (arrays[0].dtype.itemsize,)))].sum(axis=-1)
    for array in arrays[1:]: toret += popcount(array)
    return toret


def _compute_ntmp(bitweights, loc_assigned, ntile):
    """
    nbits = 64 * np.shape(bitweights)[1]
    recurr = prob_obs * nbits
    """
    # Input: list of bitweights
    nbits = 8 * sum(weight.dtype.itemsize for weight in bitweights)
    recurr = popcount(*bitweights)
    wiip = (nbits + 1) / (recurr + 1)
    zero_prob = (recurr == 0) & (~loc_assigned)
    
    #print(np.sum(zerop_msk))
    sum_ntile = np.bincount(ntile)
    sum_zero_prob = np.bincount(ntile, weights=zero_prob)
    sum_loc_assigned = np.bincount(ntile, weights=loc_assigned)
    sum_wiip = np.bincount(ntile, weights=loc_assigned * wiip)
    mask_zero_ntile = sum_ntile == 0
    frac_zero_prob = np.divide(sum_zero_prob, sum_ntile, out=np.ones_like(sum_wiip), where=~mask_zero_ntile)
    frac_missing_pw = np.divide(sum_ntile - sum_wiip, sum_ntile, out=np.ones_like(sum_wiip), where=~mask_zero_ntile)
    return frac_missing_pw, frac_zero_prob


def compute_ntmp(full_data_fn):
    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD
    ntmp = None
    if mpicomm.rank == 0:
        import fitsio
        catalog = fitsio.read(full_data_fn)
        ntmp = _compute_ntmp(_format_bitweights(catalog['BITWEIGHTS']), catalog['LOCATION_ASSIGNED'], catalog['NTILE'])
    return mpicomm.bcast(ntmp, root=0)


def _format_bitweights(bitweights):
    if bitweights.ndim == 2: return list(bitweights.T)
    return [bitweights]


def get_clustering_rdzw(*fns, kind=None, zrange=None, region=None, tracer=None, weight_type='default', ntmp=None, **kwargs):
    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD

    catalogs = [None] * len(fns)
    for ifn, fn in enumerate(fns):
        irank = ifn % mpicomm.size
        catalogs[ifn] = (irank, None)
        if mpicomm.rank == irank:  # Faster to read catalogs from one rank
            if 'fits' in str(fn):
                catalog = Catalog.read(fn, mpicomm=MPI.COMM_SELF)
            else:
                catalog = Catalog.read(fn, mpicomm=MPI.COMM_SELF, group='LSS')
            catalog.get(catalog.columns())  # Faster to read all columns at once
            columns = ['RA', 'DEC', 'Z', 'WEIGHT', 'WEIGHT_SYS', 'WEIGHT_ZFAIL', 'WEIGHT_COMP', 'WEIGHT_FKP', 'BITWEIGHTS', 'FRAC_TLOBS_TILES', 'NTILE']
            columns = [col for col in columns if col in catalog.columns()]
            catalog = catalog[columns]
            if zrange is not None:
                mask = (catalog['Z'] >= zrange[0]) & (catalog['Z'] <= zrange[1])
                catalog = catalog[mask]
            if 'bitwise' in weight_type:
                mask = (catalog['FRAC_TLOBS_TILES'] != 0)
                catalog = catalog[mask]
            if region is not None:
                mask = select_region(catalog['RA'], catalog['DEC'], region)
                catalog = catalog[mask]
            catalogs[ifn] = (irank, catalog)
    
    rdzw = []
    for irank, catalog in catalogs:
        if mpicomm.size > 1:
            catalog = Catalog.scatter(catalog, mpicomm=mpicomm, mpiroot=irank)
        individual_weight = catalog['WEIGHT']
        bitwise_weights = []
        if 'bitwise' in weight_type:
            if kind == 'data':
                individual_weight = catalog['WEIGHT'] / catalog['WEIGHT_COMP']
                bitwise_weights = _format_bitweights(catalog['BITWEIGHTS'])
            elif kind == 'randoms' and ntmp is not None:
                individual_weight = catalog['WEIGHT'] * apply_wntmp(catalog['NTILE'], ntmp)
        if 'FKP' in weight_type.upper():
            individual_weight *= catalog['WEIGHT_FKP']
        rdzw.append([catalog['RA'], catalog['DEC'], catalog['Z'], individual_weight] + bitwise_weights)
    rdzw = [np.concatenate([arrays[i] for arrays in rdzw], axis=0) for i in range(len(rdzw[0]))]
    for i in range(4): rdzw[i] = rdzw[i].astype('f8')
    return rdzw[:3], rdzw[3:]


def get_full_rdw(*fns, kind='parent', zrange=None, region=None, tracer=None, weight_type='default', ntmp=None, **kwargs):

    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD

    catalogs = [None] * len(fns)
    for ifn, fn in enumerate(fns):
        irank = ifn % mpicomm.size
        catalogs[ifn] = (irank, None)
        if mpicomm.rank == irank:  # Faster to read catalogs from one rank
            catalog = Catalog.read(fn, mpicomm=MPI.COMM_SELF, group='LSS')
            catalog.get(catalog.columns())  # Faster to read all columns at once
            columns = ['RA', 'DEC', 'LOCATION_ASSIGNED', 'BITWEIGHTS', 'NTILE', 'WEIGHT_NTILE']
            columns = [col for col in columns if col in catalog.columns()]
            catalog = catalog[columns]
            if 'fibered' in kind:
                mask = catalog['LOCATION_ASSIGNED']
                catalog = catalog[mask]
            if region is not None:
                mask = select_region(catalog['RA'], catalog['DEC'], region)
                catalog = catalog[mask]
            catalogs[ifn] = (irank, catalog)
    
    rdzw = []
    for irank, catalog in catalogs:
        if mpicomm.size > 1:
            catalog = Catalog.scatter(catalog, mpicomm=mpicomm, mpiroot=irank)
        individual_weight = catalog['WEIGHT_NTILE']
        bitwise_weights = []
        if 'fibered' in kind and 'data' in kind:
            if ntmp is not None:
                individual_weight /= apply_wntmp(catalog['NTILE'], ntmp)
            bitwise_weights = _format_bitweights(catalog['BITWEIGHTS'])
            if 'bitwise' not in weight_type:  # to be updated
                nbits = 8 * sum(weight.dtype.itemsize for weight in bitwise_weights)
                recurr = popcount(*bitwise_weights)
                wiip = (nbits + 1) / (recurr + 1)
                individual_weight *= wiip
                bitwise_weights = []
        rdzw.append([catalog['RA'], catalog['DEC'], individual_weight] + bitwise_weights)
    rdzw = [np.concatenate([arrays[i] for arrays in rdzw], axis=0) for i in range(len(rdzw[0]))]
    for i in range(3): rdzw[i] = rdzw[i].astype('f8')
    return rdzw[:2], rdzw[2:]


def get_clustering_positions_weights(*fns, **kwargs):
    from cosmoprimo.fiducial import TabulatedDESI, DESI
    fiducial = TabulatedDESI()  # faster than DESI/class (which takes ~30 s for 10 random catalogs)
    [ra, dec, z], weights = get_clustering_rdzw(*fns, **kwargs)
    dist = fiducial.comoving_radial_distance(z)
    positions = sky_to_cartesian(dist, ra, dec)
    return positions, weights


def compute_angular_upweights(output_fn, get_data, get_randoms, tracer='ELG'):
    from cucount.jax import Particles, BinAttrs, WeightAttrs, count2, setup_logging
    from lsstypes import ObservableLeaf, ObservableTree
    from lsstypes.types import Count2, Count2Correlation

    fibered_data = Particles(*get_data('fibered_data'), positions_type='rd', exchange=True)
    parent_data = Particles(*get_data('parent_data'), positions_type='rd', exchange=True)

    theta = 10**np.arange(-5, -1 + 0.1, 0.1)
    battrs = BinAttrs(theta=theta)
    wattrs = WeightAttrs(bitwise=dict(weights=fibered_data.get('bitwise_weight')))
    fibered_data_iip = fibered_data.clone(weights=wattrs(fibered_data))  # compute IIP weights

    def get_counts(*particles):
        #setup_logging('error')
        autocorr = len(particles) == 1
        weight = count2(*(particles * 2 if autocorr else particles), battrs=battrs, wattrs=wattrs)['weight']
        if autocorr:
            norm = wattrs(particles[0]).sum()**2 - wattrs(*(particles * 2)).sum()
        else:
            norm = wattrs(particles[0]).sum() * wattrs(particles[1]).sum()
        # No need to remove auto-pairs, as edges[0] > 0
        return weight / norm
        #return Count2(counts=weight, norm=norm, theta=battrs.coords('theta'), theta_edges=battrs.edges('theta'), coords=['theta'])

    DDfibered = get_counts(fibered_data)
    wattrs = WeightAttrs()
    DDparent = get_counts(parent_data)

    #parent_randoms = Particles(*get_randoms('parent_randoms'), positions_type='rd', exchange=True)
    #DRparent = get_counts(parent_data, parent_randoms)
    #DRfibered = get_counts(fibered_data_iip, parent_randoms)
    kw = dict(theta=battrs.coords('theta'), theta_edges=battrs.edges('theta'), coords=['theta'])
    auw = {}
    auw['DD'] = ObservableLeaf(value=np.where(DDfibered == 0., 1., DDparent / DDfibered), **kw)
    #auw['DR'] = ObservableLeaf(value=np.where(DRfibered == 0., 1., DRparent / DRfibered), **kw)
    auw = ObservableTree(list(auw.values()), pairs=list(auw.keys()))
    if output_fn is not None and jax.process_index() == 0:
        logger.info(f'Writing to {output_fn}')
        auw.write(output_fn)
    return auw


def compute_fkp_effective_redshift(fkp, cellsize=10., order=2):
    from jax import numpy as jnp
    from cosmoprimo.fiducial import TabulatedDESI, DESI
    from cosmoprimo.utils import DistanceToRedshift
    from jaxpower import compute_fkp2_normalization, compute_fkp3_normalization, FKPField
    fiducial = TabulatedDESI()
    d2z = DistanceToRedshift(lambda z: jnp.array(fiducial.comoving_radial_distance(z)))

    compute_fkp_normalization = {2: compute_fkp2_normalization, 3: compute_fkp3_normalization}[order]

    def compute_z(positions):
        return d2z(jnp.sqrt(jnp.sum(positions**2, axis=-1)))

    if isinstance(fkp, FKPField):
        norm = compute_fkp_normalization(fkp, cellsize=cellsize)
        fkp = fkp.clone(data=fkp.data.clone(weights=data.weights * compute_z(fkp.data.positions)), randoms=randoms.clone(weights=fkp.randoms.weights  * compute_z(fkp.randoms.positions)))
        znorm = compute_fkp_normalization(fkp, cellsize=cellsize)
    else:  # fkp is randoms
        norm = compute_fkp_normalization(fkp, cellsize=cellsize, split=42)
        fkp = fkp.clone(weights=fkp.weights * compute_z(fkp.positions))
        znorm = compute_fkp_normalization(fkp, cellsize=cellsize, split=42)
    return znorm / norm

    
def combine_regions(output_fn, fns, logger=None):
    combined = types.sum([types.read(fn) for fn in fns])  # for the covariance matrix, assumes observables are independent
    if output_fn is not None:
        if logger: logger.info(f'Writing to {output_fn}')
        combined.write(output_fn)
    #return combined