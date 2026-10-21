import numpy as np

def estimate_deflection_gpu_memory(B, Nx, Ny, nfw_cse=16, hern_cse=13, dtype_bytes=4):
    """
    Estimate GPU memory usage for the fully vectorized deflection code.

    Parameters
    ----------
    B : int
        Number of galaxies (batch size)
    Nx : int
        Number of x-grid points
    Ny : int
        Number of y-grid points
    nfw_cse : int
        Number of NFW CSE grids (default 16)
    hern_cse : int
        Number of Hernquist CSE grids (default 13)
    dtype_bytes : int
        Size of data type in bytes (float32 = 4)

    Returns
    -------
    memory_gb : float
        Estimated GPU memory usage in GB
    """
    # Main gradient arrays
    grad_memory = B * Nx * Ny * dtype_bytes * 2  # gradx + grady

    # CSE arrays for NFW and Hernquist
    # Each temporary array: shape (B, CSE, Ny, Nx)
    nfw_memory = B * nfw_cse * Nx * Ny * dtype_bytes * 2  # Xr_cse + Yr_cse
    hern_memory = B * hern_cse * Nx * Ny * dtype_bytes * 2  # Xr_cse + Yr_cse

    # Total estimated memory (GB)
    total_bytes = grad_memory + nfw_memory + hern_memory
    memory_gb = total_bytes / 1024**3

    # Add ~20% overhead for intermediate arrays and multipoles/external shear
    memory_gb *= 1.2

    return memory_gb

nph = 50
Nx = Ny = 2 * nph  # grid points in each direction
B = 5000           # number of galaxies

estimated_gb = estimate_deflection_gpu_memory(B, Nx, Ny)
print(f"Estimated GPU memory for B={B}: {estimated_gb:.3f} GB")



"""

    # Ensure we're on GPU 0
    cp.cuda.Device(0).use()
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()  # clear previous allocations
    
    def print_gpu_memory(tag=""):
        used = mempool.used_bytes() / 1024**3  # in GB
        total = cp.cuda.runtime.memGetInfo()[1] / 1024**3  # total GPU memory in GB
        print(f"[{tag}] GPU memory used: {used:.3f} GB, total: {total:.3f} GB")
        
    # Before calculation
    print_gpu_memory("before")

    start = time.time()
    
    # After calculation
    print_gpu_memory("after")

    end = time.time()
    print(f"Elapsed time: {end-start:.3f} s")


"""

