from momi3.momi import Momi3
import jax.random as jr
import jax.numpy as jnp
import jax
from momi3.jsfs import JSFS

def initialize_momi_iicr(demo, params):
    momi_object = Momi3(demo).iicr(2)
    f, x = momi_object.reparameterize(list(params))
    parameters = list(x.keys())
    return momi_object, f, x, parameters

def initialize_momi_sfs(ts, deme_model):
    pop_sample_counts = {
        pop.metadata['name']: ts.samples(population=pop.id).shape[0]
        for pop in ts.populations()
        if ts.samples(population=pop.id).shape[0] != 0  # Only include if sample count > 0
    }
    sample_ids = [ts.samples(population=pop.id) 
              for pop in ts.populations()
              if ts.samples(population=pop.id).size != 0
              ]
    momi_sfs_object = Momi3(deme_model).sfs(pop_sample_counts)
    afs = ts.allele_frequency_spectrum(sample_sets=sample_ids, span_normalise=False)
    jsfs = JSFS.from_dense(afs, list(pop_sample_counts))
    
    return momi_sfs_object, jsfs

def sample_tmrca_spanss(ts, subkey=jax.random.PRNGKey(1), num_pop=2):
    sample_config = {f"P{i}": 0 for i in range(num_pop)}
    samples = jax.random.choice(subkey, ts.num_samples, shape=(2,), replace=False)
    sample1, sample2 = samples[0], samples[1]

    pop1 = ts.node(sample1.item(0)).population - 1
    pop2 = ts.node(sample2.item(0)).population - 1
    sample_config[f"P{pop1}"] += 1
    sample_config[f"P{pop2}"] += 1

    # Precompute all TMRCAs and spans into arrays
    tmrcas = []
    spans = []
    for tree in ts.trees():
        spans.append(tree.interval.right - tree.interval.left)
        tmrcas.append(tree.tmrca(sample1, sample2))
    
    # Convert to JAX arrays
    tmrcas = jnp.array(tmrcas)  # Shape: (num_trees,)
    spans = jnp.array(spans)    # Shape: (num_trees,)
    tmrcas_spans = jnp.stack([tmrcas, spans], axis=1)  # Shape: (num_trees, 2)

    # Merge consecutive spans with same TMRCA
    def merge_spans(carry, x):
        current_tmrca, current_span, idx, output = carry
        tmrca, span = x
        
        # Update each component individually
        new_tmrca = jnp.where(tmrca == current_tmrca, current_tmrca, tmrca)
        new_span = jnp.where(tmrca == current_tmrca, current_span + span, span)
        new_idx = jnp.where(tmrca == current_tmrca, idx, idx + 1)
        new_output = jnp.where(
            tmrca == current_tmrca, 
            output, 
            output.at[idx].set(jnp.array([current_tmrca, current_span]))
        )
        
        return (new_tmrca, new_span, new_idx, new_output), None

    init_carry = (tmrcas_spans[0, 0], 0.0, 0, jnp.full((ts.num_trees, 2), jnp.array([1.0, 0.0])))
    final_carry, _ = jax.lax.scan(merge_spans, init_carry, tmrcas_spans)
    final_tmrca, final_span, _, final_output = final_carry
    final_output = final_output.at[-1].set(jnp.array([final_tmrca, final_span]))
    is_ones = jnp.all(final_output == jnp.array([1.0, 0.0]), axis=1)
    reordered_arr = jnp.concatenate([final_output[~is_ones], final_output[is_ones]])

    return reordered_arr, sample_config

import jax
import jax.numpy as jnp
import phlash
from phlash.likelihood.arg import log_density

def call_momi(t_value, num_samples, x, f, momi_object):
    """Computes c and p from momi_object."""
    c, p = momi_object(t=t_value, num_samples=num_samples, params=f(x))
    return c, p

# Vectorizing over t
batched_call_momi = jax.vmap(call_momi, in_axes=(0, None, None, None, None))

def iicr_likelihood(x, tmrca_spans, num_samples, max_tmrca, f, momi_object):
    """Computes the negative log-likelihood (for minimization)."""
    # t = jnp.linspace(1e-4, jnp.max(tmrca_spans[:, 0]).item(), 5000)
    t = jnp.linspace(1e-4, max_tmrca, 5000)
    c_values, log_p_values = batched_call_momi(t, num_samples, x, f, momi_object)
    eta = phlash.size_history.SizeHistory(t=t, c=c_values)
    dm = phlash.size_history.DemographicModel(
        eta=eta, theta=None, rho=1e-8
    )
    return log_density(dm, tmrca_spans[None])

def finite_diff_hessian_iicr(likelihood, parameters, x, values, tmrca_spans, sample_config, f, momi_object, eps=1e-5):
    grad_fn = jax.grad(likelihood)
    n = len(parameters)
    H = jnp.zeros((n, n))
    updated_x = x.copy()

    for j in range(n):
        updated_x[parameters[j]] = values[j]
    
    for i in range(n):
        params_plus = updated_x.copy()
        params_minus = updated_x.copy()
        params_plus[parameters[i]] = values[i]+eps
        params_minus[parameters[i]] = values[i]-eps
        max_tmrca = jnp.max(tmrca_spans[:, 0]).item()

        grad_plus = grad_fn(params_plus, tmrca_spans, sample_config, max_tmrca, f, momi_object)
        grad_minus = grad_fn(params_minus, tmrca_spans, sample_config, max_tmrca, f, momi_object)
        H = H.at[i,:].set((grad_plus[parameters[i]] - grad_minus[parameters[i]]) / (2*eps))
    
    # Symmetrize
    return (H + H.T) / 2 

def sfs_likelihood(params, momi_sfs_object, jsfs):
    return momi_sfs_object.loglik(params, jsfs)

def finite_diff_hessian_sfs(likelihood, parameters, x, values, momi_sfs_object, jsfs, eps=1e-5):
    grad_fn = jax.grad(likelihood)
    n = len(x)
    H = jnp.zeros((n, n))
    updated_x = x.copy()

    for j in range(n):
        updated_x[parameters[j]] = values[j]
    
    for i in range(n):
        params_plus = updated_x.copy()
        params_minus = updated_x.copy()
        params_plus[parameters[i]] = values[i]+eps
        params_minus[parameters[i]] = values[i]-eps
        print(params_plus)
        print(params_minus)

        grad_plus = grad_fn(params_plus, momi_sfs_object, jsfs)
        grad_minus = grad_fn(params_minus, momi_sfs_object, jsfs)
        H = H.at[i,:].set((grad_plus[parameters[i]] - grad_minus[parameters[i]]) / (2*eps))
    
    # Symmetrize
    return (H + H.T) / 2


def hessian_approximation(demo, ts, params, type, values, seed = 0):
    momi_object, f, x, parameters = initialize_momi_iicr(demo, params)
    key = jr.PRNGKey(seed)
    results = []

    if type not in ("IICR", "SFS", "BOTH"):
        print("Invalid type. Expected 'IICR' or 'SFS' (case sensitive)")

    if type == "IICR" or type == "BOTH":
        print("hi")
        key, subkey = jr.split(key)
        tmrca_span, sample_config = sample_tmrca_spanss(ts, subkey)
        hessian_iicr = finite_diff_hessian_iicr(iicr_likelihood, parameters, x, values, tmrca_span, sample_config, f, momi_object)
        inv_hessian_iicr = jnp.linalg.inv(-hessian_iicr)
        results.append(hessian_iicr)
        results.append(inv_hessian_iicr)

    if type == "SFS" or type == "BOTH":
        print("hi")
        momi_sfs_object, jsfs = initialize_momi_sfs(ts, demo)
        hessian_iicr = finite_diff_hessian_sfs(sfs_likelihood, parameters, x, values, momi_sfs_object, jsfs)
        inv_hessian_iicr = jnp.linalg.inv(-hessian_iicr)
        results.append(hessian_iicr)
        results.append(inv_hessian_iicr)

    return results