import math
from opacus.accountants.utils import get_noise_multiplier
def compute_client_sigmas(loaders, args, accountant="gdp"):
    """Return a list sigma[i] for each client loader."""
    sigmas = []
    for loader in loaders:
        N_i = len(loader.dataset)  # exact because we built from a Python list
        B_i = loader.batch_size    # this is the *logical* batch (with BMM if you use it)
        # steps per epoch: exact if drop_last=True; else Conservative ceil(...)
        total_steps = args.rounds * args.local_steps
        # sample rate
        q_i = B_i / N_i
        sigma_i = get_noise_multiplier(
            target_epsilon=args.epsilon,
            target_delta=1/N_i**1.1,  # a common choice of delta
            sample_rate=q_i,
            steps=total_steps,
            accountant=accountant,  # "rdp" (faster) or "prv" (tighter)
        )
        sigmas.append(sigma_i)
    return sigmas