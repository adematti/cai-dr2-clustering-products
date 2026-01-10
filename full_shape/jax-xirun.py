def compute_pycorr_particle2_correlation(output_fn, get_data, get_randoms, auw=None, cut=None, engine='corrfunc', gpu=True, nthreads=4):
    from pycorr import TwoPointCorrelationFunction, setup_logging
    from lsstypes.external import from_pycorr

    edges = (np.linspace(0., 180., 181), np.linspace(-1., 1., 201))
    data_positions, data_weights = get_data()
    with_bitwise = len(data_weights) > 1
    weight_attrs, D1D2_twopoint_weights = {}, None
    if with_bitwise:
        weight_attrs['normalization'] = 'counter'
    if auw is not None:
        D1D2_twopoint_weights = (auw.get('DD').coords('theta'), auw.get('DD').value())
    
    D1D2 = None
    correlation = 0
    setup_logging('debug')
    for iran, (randoms_positions, randoms_weights) in enumerate(get_randoms()):
        logger.info(f'Processing random {iran:d}')
        correlation += TwoPointCorrelationFunction(mode='smu', edges=edges, data_positions1=data_positions, data_weights1=data_weights,
                                                   randoms_positions1=randoms_positions, randoms_weights1=randoms_weights,
                                                   position_type='pos', los='midpoint', weight_attrs=weight_attrs,
                                                   D1D2_twopoint_weights=D1D2_twopoint_weights, D1D2=D1D2, engine=engine, gpu=gpu, nthreads=nthreads)

    correlation = from_pycorr(correlation)
    if output_fn is not None:
        logger.info(f'Writing to {output_fn}')
        correlation.write(output_fn)
    return correlation



def compute_cucount_particle2_correlation(output_fn, get_data, get_randoms, auw=None, cut=None):
    from cucount.jax import Particles, BinAttrs, WeightAttrs, count2, setup_logging
    import lsstypes as types
    from lsstypes import Count2, Count2Correlation

    data = Particles(*get_data(), exchange=True)

    battrs = BinAttrs(s=np.linspace(0., 180., 181), mu=(np.linspace(-1., 1., 201), 'midpoint'))
    bitwise = angular = None
    if data.get('bitwise_weight'):
        bitwise = dict(weights=data.get('bitwise_weight'))
    if auw is not None:
        angular = dict(sep=auw.get('DD').coords('theta'), weight=auw.get('DD').value())
    wattrs = WeightAttrs(bitwise=bitwise, angular=angular)
    mattrs = None  # automatic setting for mesh
    
    # Helper to convert to lsstypes Count2
    def to_lsstypes(battrs: BinAttrs, counts: np.ndarray, norm: np.ndarray) -> Count2:
        coords = battrs.coords()
        edges = battrs.edges()
        edges = {f'{k}_edges': v for k, v in edges.items()}
        return Count2(counts=counts, norm=norm * np.ones_like(counts), **coords, **edges, coords=list(coords))
    
    # Hepler to get counts as Count2
    def get_counts(*particles: Particles, wattrs: WeightAttrs=None) -> Count2:
        if wattrs is None: wattrs = WeightAttrs()
        autocorr = len(particles) == 1
        counts = count2(*(particles * 2 if autocorr else particles), battrs=battrs, wattrs=wattrs, mattrs=mattrs)['weight']
        if autocorr:
            auto_sum = wattrs(*(particles * 2)).sum()
            norm = wattrs(particles[0]).sum()**2 - auto_sum
            # Correct auto-pairs
            zero_index = tuple(np.flatnonzero((0 >= edges[:, 0]) & (0 < edges[:, 1])) for edges in battrs.edges().values())
            counts = counts.at[zero_index].add(-auto_sum)
        else:
            norm = wattrs(particles[0]).sum() * wattrs(particles[1]).sum()
        return to_lsstypes(battrs, counts, norm)
    
    DD = get_counts(data, wattrs=wattrs)
    data = data.clone(weights=wattrs(data))  # clone data, with IIP weights (in case we provided bitwise weights)
    DR, RR = [], []
    for iran, randoms in enumerate(get_randoms()):
        if jax.process_index() == 0:
            logger.info(f'Processing random {iran:d}')
        randoms = Particles(*randoms, exchange=True)
        DR.append(get_counts(data, randoms))
        RR.append(get_counts(randoms))
    DR = types.sum(DR)
    RR = types.sum(RR)

    RD = DR.clone(value=DR.value()[:, ::-1])  # reverse mu for RD

    # For reconstructed 2PCF, you can provide DD, DS, SD, SS, RR counts
    correlation = Count2Correlation(estimator='landyszalay', DD=DD, DR=DR, RD=RD, RR=RR)
    if output_fn is not None and jax.process_index() == 0:
        logger.info(f'Writing to {output_fn}')
        correlation.write(output_fn)
    return correlation


def compute_cucount_RR2(output_fn, get_randoms):
    from cucount.jax import Particles, BinAttrs, WeightAttrs, count2, setup_logging
    import lsstypes as types
    from lsstypes import Count2

    battrs = BinAttrs(s=np.linspace(0., 180., 181), mu=(np.linspace(-1., 1., 201), 'midpoint'))
    mattrs = None

    # Helper to convert to lsstypes Count2
    def to_lsstypes(battrs: BinAttrs, counts: np.ndarray, norm: np.ndarray) -> Count2:
        coords = battrs.coords()
        edges = battrs.edges()
        edges = {f'{k}_edges': v for k, v in edges.items()}
        return Count2(counts=counts, norm=norm * np.ones_like(counts), **coords, **edges, coords=list(coords))
    
    # Hepler to get counts as Count2
    def get_counts(*particles: Particles, wattrs: WeightAttrs=None) -> Count2:
        if wattrs is None: wattrs = WeightAttrs()
        autocorr = len(particles) == 1
        counts = count2(*(particles * 2 if autocorr else particles), battrs=battrs, wattrs=wattrs, mattrs=mattrs)['weight']
        if autocorr:
            auto_sum = wattrs(*(particles * 2)).sum()
            norm = wattrs(particles[0]).sum()**2 - auto_sum
            # Correct auto-pairs
            zero_index = tuple(np.flatnonzero((0 >= edges[:, 0]) & (0 < edges[:, 1])) for edges in battrs.edges().values())
            counts = counts.at[zero_index].add(-auto_sum)
        else:
            norm = wattrs(particles[0]).sum() * wattrs(particles[1]).sum()
        return to_lsstypes(battrs, counts, norm)

    RR, norm_fkp = [], []
    for iran, _randoms in enumerate(get_randoms()):
        if jax.process_index() == 0:
            logger.info(f'Processing random {iran:d}')
        randoms = Particles(*_randoms, exchange=True)
        RR.append(get_counts(randoms))
        from jaxpower import ParticleField, compute_fkp2_normalization, get_mesh_attrs
        randoms = ParticleField(_randoms[0], _randoms[1][0], attrs=get_mesh_attrs(_randoms[0], cellsize=10., check=True), exchange=True, backend='jax')
        norm_fkp.append(compute_fkp2_normalization(randoms, split=42))
    RR = types.sum(RR)
    RR.attrs['norm_fkp'] = sum(norm_fkp)
    if output_fn is not None and jax.process_index() == 0:
        logger.info(f'Writing to {output_fn}')
        RR.write(output_fn)
    return RR