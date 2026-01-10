import time
import logging

import jax
import numpy as np
import lsstypes as types

logger = logging.getLogger('summary-statistics')


def compute_angular_upweights(*get_data, output_fn=None):
    from cucount.jax import Particles, BinAttrs, WeightAttrs, count2, setup_logging
    from lsstypes import ObservableLeaf, ObservableTree

    all_fibered_data, all_parent_data = [], []
    for _get_data in get_data:
        fibered_data, parent_data = _get_data()
        fibered_data = Particles(*fibered_data, positions_type='rd', exchange=True)
        parent_data = Particles(*parent_data, positions_type='rd', exchange=True)
        all_fibered_data.append(fibered_data)
        all_parent_data.append(parent_data)

    theta = 10**np.arange(-5, -1 + 0.1, 0.1)
    battrs = BinAttrs(theta=theta)
    wattrs = WeightAttrs(bitwise=dict(weights=fibered_data.get('bitwise_weight')))

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

    DDfibered = get_counts(*all_fibered_data)
    wattrs = WeightAttrs()
    DDparent = get_counts(*all_parent_data)

    kw = dict(theta=battrs.coords('theta'), theta_edges=battrs.edges('theta'), coords=['theta'])
    auw = {}
    auw['DD'] = ObservableLeaf(value=np.where(DDfibered == 0., 1., DDparent / DDfibered), **kw)
    #auw['DR'] = ObservableLeaf(value=np.where(DRfibered == 0., 1., DRparent / DRfibered), **kw)
    auw = ObservableTree(list(auw.values()), pairs=list(auw.keys()))
    if output_fn is not None and jax.process_index() == 0:
        logger.info(f'Writing to {output_fn}')
        auw.write(output_fn)
    return auw


def prepare_jaxpower_particles(*get_data_randoms, mattrs=None):
    from jaxpower import get_mesh_attrs, ParticleField

    all_data, all_randoms, all_shifted = [], [], []
    for _get_data_randoms in get_data_randoms:
        # data, randoms (optionally shifted) are tuples (positions, weights)
        data, randoms, *shifted = _get_data_randoms()
        all_data.append(data)
        all_randoms.append(randoms)
        if shifted:
            all_shifted.append(shifted)

    assert len(all_shifted) == len(data), 'Give as many shifted randoms as data/randoms'

    # Define the mesh attributes; pass in positions only
    mattrs = get_mesh_attrs(*[data[0] for data in all_data + all_shifted + all_randoms], check=True, **(mattrs or {}))

    all_particles = []
    for i, (data, randoms) in enumerate(zip(all_data, all_randoms)):
        data = ParticleField(*data, attrs=mattrs, exchange=True, backend='mpi')
        randoms = ParticleField(*randoms, attrs=mattrs, exchange=True, backend='mpi')
        if all_shifted:
            shifted = ParticleField(*shifted, attrs=mattrs, exchange=True, backend='mpi')
        else:
            shifted = None
        all_particles.append((data, randoms, shifted))
    if jax.process_index() == 0:
        logger.info(f'All particles on the GPU.')

    return all_particles


def _get_jaxpower_attrs(*particles):
    mattrs = particles[0].attrs
    # Creating FKP fields
    attrs = {name: mattrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']}
    for i, (data, randoms, shifted) in enumerate(particles):
        attrs[f'size_data{i:d}'], attrs[f'wsum_data{i:d}'] = data.size, data.sum()
        attrs[f'size_randoms{i:d}'], attrs[f'wsum_randoms{i:d}'] = randoms.size, randoms.sum()
        if shifted is not None:
            attrs[f'size_shifted{i:d}'], attrs[f'wsum_shifted{i:d}'] = shifted.size, shifted.sum()
    return attrs


def compute_mesh2_spectrum(*particles, output_fn=None, auw=None, cut=None,
                           ells=(0, 2, 4), los='firstpoint', cache=None):

    from jaxpower import (FKPField, compute_fkp2_normalization, compute_fkp2_shotnoise, BinMesh2SpectrumPoles, compute_mesh2_spectrum,
                          BinParticle2SpectrumPoles, BinParticle2CorrelationPoles, compute_particle2, compute_particle2_shotnoise)

    attrs = _get_jaxpower_attrs(*particles)
    mattrs = particles[0][0].attrs
    # Define the binner
    if cache is None: cache = {}
    bin = cache.get('bin_mesh2_spectrum', None)
    if bin is None: bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.001}, ells=ells)
    cache.setdefault('bin_mesh2_spectrum', bin)

    # Computing normalization
    all_fkp = [FKPField(data, randoms) for (data, randoms, _) in particles]
    norm = compute_fkp2_normalization(*all_fkp, bin=bin, cellsize=10)

    # Computing shot noise
    all_fkp = [FKPField(data, shifted) for (data, _, shifted) in particles]
    num_shotnoise = compute_fkp2_shotnoise(*all_fkp, bin=bin)

    jitted_compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'], donate_argnums=[0])
    #jitted_compute_mesh2_spectrum = compute_mesh2_spectrum
    kw = dict(resampler='tsc', interlacing=3, compensate=True)
    spectrum = jitted_compute_mesh2_spectrum(*[fkp.paint(**kw, out='complex') for fkp in all_fkp], bin=bin, los=los)
    spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise, attrs=attrs)

    jax.block_until_ready(spectrum)
    if jax.process_index() == 0:
        logger.info('Mesh-based computation finished')

    if cut is not None:
        assert auw is None, 'angular cut and angular upweighting are mutually exclusive'
        sattrs = {'theta': (0., 0.05)}
        #bin = BinParticle2SpectrumPoles(mattrs, edges=bin.edges, xavg=bin.xavg, sattrs=sattrs, ells=ells)
        bin = BinParticle2CorrelationPoles(mattrs, edges={'step': 0.1}, sattrs=sattrs, ells=ells)
        from jaxpower.particle2 import convert_particles
        all_particles = [convert_particles(fkp.particles) for fkp in all_fkp]
        close = compute_particle2(*all_particles, bin=bin, los=los)
        close = close.clone(num_shotnoise=compute_particle2_shotnoise(*all_particles, bin=bin), norm=norm)
        close = close.to_spectrum(spectrum)
        spectrum = spectrum.clone(value=spectrum.value() - close.value())

    if auw is not None:
        from cucount.jax import WeightAttrs
        from jaxpower.particle2 import convert_particles
        sattrs = {'theta': (0., 0.1)}
        all_data = [convert_particles(fkp.data, weights=[fkp.weights] * 2, index_value=dict(individual_weight=1, negative_weight=1)) for fkp in all_fkp]
        wattrs = WeightAttrs(angular=dict(sep=auw.get('DD').coords('theta'), weight=auw.get('DD').value()) if auw is not None else None)
        bin = BinParticle2SpectrumPoles(mattrs, edges=bin.edges, xavg=bin.xavg, sattrs=sattrs, wattrs=wattrs, ells=ells)
        DD = compute_particle2(*all_data, bin=bin, los=los)
        DD = DD.clone(num_shotnoise=compute_particle2_shotnoise(*all_data, bin=bin), norm=norm)
        spectrum = spectrum.clone(value=spectrum.value() + DD.value())

    jax.block_until_ready(spectrum)
    if jax.process_index() == 0:
        logger.info(f'Particle-based calculation finished.')

    if output_fn is not None and jax.process_index() == 0:
        logger.info(f'Writing to {output_fn}')
        spectrum.write(output_fn)

    return spectrum


def compute_jaxpower_mesh3_spectrum(*particles, output_fn=None,
                                    basis='scoccimarro', ells=[0, 2], los='local', mattrs=None,
                                    buffer_size=0, cache=None):
    from jaxpower import (FKPField, compute_fkp3_normalization, compute_fkp3_shotnoise, BinMesh3SpectrumPoles, compute_mesh3_spectrum)

    attrs = _get_jaxpower_attrs(*particles)
    mattrs = particles[0][0].attrs
    # Define the binner
    if cache is None: cache = {}
    bin = cache.get('bin_mesh3_spectrum', None)
    if bin is None: bin = BinMesh3SpectrumPoles(mattrs, edges={'step': 0.01 if 'scoccimarro' in basis else 0.005}, basis=basis, ells=ells, buffer_size=buffer_size)
    cache.setdefault('bin_mesh3_spectrum', bin)

    # Computing normalization
    all_fkp = [FKPField(data, randoms) for (data, randoms, _) in particles]
    norm = compute_fkp3_normalization(*all_fkp, bin=bin, split=42, cellsize=10)

    # Computing shot noise
    all_fkp = [FKPField(data, shifted) for (data, _, shifted) in particles]
    num_shotnoise = compute_fkp3_shotnoise(*all_fkp, bin=bin)

    kw = dict(resampler='tsc', interlacing=3, compensate=True)
    num_shotnoise = compute_fkp3_shotnoise(*all_fkp, los=los, bin=bin, **kw)
    jitted_compute_mesh3_spectrum = jax.jit(compute_mesh3_spectrum, static_argnames=['los'], donate_argnums=[0])

    spectrum = jitted_compute_mesh3_spectrum(*[fkp.paint(**kw, out='complex') for fkp in all_fkp], los=los, bin=bin)
    spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)

    spectrum = spectrum.clone(attrs=attrs)
    if output_fn is not None and jax.process_index() == 0:
        logger.info(f'Writing to {output_fn}')
        spectrum.write(output_fn)
    return spectrum


def prepare_cucount_particles(*get_data_randoms, mattrs=None):
    from cucount.jax import Particles

    all_data, all_randoms, all_shifted = [], [], []
    for _get_data_randoms in get_data_randoms:
        # data, randoms (optionally shifted) are tuples (positions, weights)
        data, randoms, *shifted = _get_data_randoms()
        all_data.append(data)
        all_randoms.append(randoms)
        if shifted:
            all_shifted.append(shifted)

    assert len(all_shifted) == len(data), 'Give as many shifted randoms as data/randoms'

    def get_all_particles(particles):
        if isinstance(particles, tuple) and not isinstance(particles[0], tuple):  # positions, weights
            return [Particles(*particles, exchange=True)]
        else:
            return (Particles(*pp, exchange=True) for pp in particles)

    all_particles = []
    for i, (data, randoms) in enumerate(zip(all_data, all_randoms)):
        data = Particles(*data, exchange=True)
        randoms = get_all_particles(randoms)
        if all_shifted:
            shifted = get_all_particles(shifted)
        else:
            shifted = None
        all_particles.append((data, randoms, shifted))
    if jax.process_index() == 0:
        logger.info(f'All particles on the GPU.')

    return all_particles


def compute_particle2_correlation(*particles, output_fn=None, auw=None, cut=None):
    from cucount.jax import Particles, BinAttrs, WeightAttrs, SelectionAttrs, MeshAttrs, count2, setup_logging
    from lsstypes import Count2, Count2Correlation

    battrs = BinAttrs(s=np.linspace(0., 180., 181), mu=(np.linspace(-1., 1., 201), 'midpoint'))
    #battrs = BinAttrs(s=np.linspace(0., 150., 151), mu=(np.linspace(-1., 1., 201), 'midpoint'))
    bitwise = angular = None
    if data.get('bitwise_weight'):
        bitwise = dict(weights=data.get('bitwise_weight'))
    if cut is not None:
        sattrs = SelectionAttrs(theta=(0., 0.05))
    if auw is not None:
        angular = dict(sep=auw.get('DD').coords('theta'), weight=auw.get('DD').value())
    wattrs = WeightAttrs(bitwise=bitwise, angular=angular)
    mattrs = None  # automatic setting for mesh

    # Helper to convert to lsstypes Count2
    def to_lsstypes(battrs: BinAttrs, counts: np.ndarray, norm: np.ndarray, attrs: dict) -> Count2:
        coords = battrs.coords()
        edges = battrs.edges()
        edges = {f'{k}_edges': v for k, v in edges.items()}
        return Count2(counts=counts, norm=norm * np.ones_like(counts), **coords, **edges, coords=list(coords), attrs=attrs)

    # Hepler to get counts as Count2
    def get_counts(*particles: Particles, wattrs: WeightAttrs=None) -> Count2:
        if wattrs is None: wattrs = WeightAttrs()
        if sattrs is None: sattrs = SelectionAttrs()
        autocorr = len(particles) == 1
        counts = count2(*(particles * 2 if autocorr else particles), battrs=battrs, wattrs=wattrs, mattrs=mattrs, sattrs=sattrs)['weight']
        attrs = {'size1': len(particles[0]), 'wsum1': wattrs(particles[0]).sum()}
        if autocorr:
            auto_sum = wattrs(*(particles * 2)).sum()
            norm = wattrs(particles[0]).sum()**2 - auto_sum
            # Correct auto-pairs
            zero_index = tuple(np.flatnonzero((0 >= edges[:, 0]) & (0 < edges[:, 1])) for edges in battrs.edges().values())
            counts = counts.at[zero_index].add(-auto_sum)
        else:
            norm = wattrs(particles[0]).sum() * wattrs(particles[1]).sum()
            attrs.update({'size2': len(particles[0]), 'wsum2': wattrs(particles[0]).sum()})
        return to_lsstypes(battrs, counts, norm, attrs=attrs)

    all_data, all_randoms, all_shifted = tuple(zip(*particles))

    DD = get_counts(*all_data, wattrs=wattrs)
    data = data.clone(weights=wattrs(data))  # clone data, with IIP weights (in case we provided bitwise weights)

    DS, SD, SS, RR = [], [], []
    iran = 0
    for all_randoms, all_shifted in zip(zip(*all_randoms, strict=True), zip(*all_shifted, strict=True), strict=True):
        if jax.process_index() == 0:
            logger.info(f'Processing random {iran:d}.')
        iran += 1
        RR.append(get_counts(*all_randoms))
        if all(shifted is not None for shifted in all_shifted):
            SS.append(get_counts(*all_shifted))
        else:
            all_shifted = all_randoms
            SS.append(RR[-1])
        DS.append(get_counts(all_data[0], all_shifted[1]))
        SD.append(get_counts(all_shifted[0], all_data[1]))

    DS, SD, SS, RR = (types.sum(XX) for XX in [DS, SD, SS, RR])
    correlation = Count2Correlation(estimator='landyszalay', DD=DD, DS=DS, SD=SD, SS=SS, RR=RR)
    if output_fn is not None and jax.process_index() == 0:
        logger.info(f'Writing to {output_fn}')
        correlation.write(output_fn)

    return correlation



def compute_summary_statistics_from_filename(stats=tuple(), regions='NGC', zrange=None):

    # Here do the optimal calculation given stats
