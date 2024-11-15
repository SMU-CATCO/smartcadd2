import jax
import jax.numpy as jnp
import e3nn_jax as e3nn

def shifted_softplus(x: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.softplus(x) - jnp.log(2.0)


def normalized_bessel(d: jnp.ndarray, n: int) -> jnp.ndarray:
    with jax.ensure_compile_time_eval():
        r = jnp.linspace(0.0, 1.0, 1000, dtype=d.dtype)
        b = e3nn.bessel(r, n)
        mu = jax.scipy.integrate.trapezoid(b, r, axis=0)
        sig = jax.scipy.integrate.trapezoid((b - mu) ** 2, r, axis=0) ** 0.5
    return (e3nn.bessel(d, n) - mu) / sig


def u(d: jnp.ndarray, p: int) -> jnp.ndarray:
    return e3nn.poly_envelope(p - 1, 2)(d)
