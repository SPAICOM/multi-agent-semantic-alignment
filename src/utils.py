"""In this module there are defined usefull methods:

- complex_compressed_tensor:
    A method used to compress a real tensor. The features are encoded half as real and half as imaginary.
- decompress_complex_tensor:
    The function decompress the complex compressed tensor in the original real domain.
- complex_tensor:
    Get the complex form of a tensor.
- complex_gaussian_matrix:
    A method that returns a complex gaussian matrix in the torch.Tensor format.
- awgn:
    A function that returns a noise vector sampled by a complex gaussian of a specified sigma.
- snr:
    Return the Signal to Noise Ratio.
- sigma_given_snr:
    Given a fixed value of SNR and signal, retrieve the correspoding value of sigma.
- a_inv_times_b:
    Perform in an efficient way the A^{-1}B.
- prewhiten:
    Prewhiten the training and test data using only training data statistics.
- remove_non_empty_dir:
    A function to remove a non-empty folder. Use with caution.
- mmse_svd_equalizer:
    A function to perform SVD channel equalization.
"""

import math
import torch
import shutil
from pathlib import Path


# ================================================================
#
#                        Methods Definition
#
# ================================================================


def complex_compressed_tensor(
    x: torch.Tensor,
    device: str = 'cpu',
) -> torch.Tensor:
    """Converts a real-valued tensor into a complex-valued tensor by compressing
    the feature dimension. The first half of the original feature dimension
    is used as the real part, and the second half is used as the imaginary part.

    If the feature dimension (d) is odd, an extra row of zeros is appended
    to make it even before splitting.

    Args:
        x : torch.Tensor
            Input tensor of shape (d, n), where d is the feature dimension
            and n is the number of samples.
        device : str, optional
            The device to which the output tensor should be moved. Default is "cpu".

        Returns:
            torch.Tensor
                A complex-valued tensor of shape (d/2, n) where each entry is a
                complex number constructed from the original real-valued tensor.
    """
    d, n = x.shape

    if d % 2 != 0:
        x = torch.cat(
            (x, torch.zeros((1, n), dtype=x.dtype, device=x.device)), dim=0
        )

        # Ensuring even dimension for splitting
        d += 1

    real_part = x[: d // 2, :]
    imaginary_part = x[d // 2 :, :]

    # Combine real and imaginary parts into a complex tensor
    x = torch.stack((real_part, imaginary_part), dim=-1)

    return torch.view_as_complex(x).to(device)


def decompress_complex_tensor(
    x: torch.Tensor,
    device: str = 'cpu',
) -> torch.Tensor:
    """Decompresses a complex-valued tensor back into its original real-valued form.
    The function separates the real and imaginary parts and concatenates them
    along the feature dimension.

    Args:
        x : torch.Tensor
            Input complex-valued tensor of shape (d/2, n), where each entry
            is a complex number.
        device : str, optional
            The device to which the output tensor should be moved. Default is "cpu".

    Returns:
        torch.Tensor
            A real-valued tensor of shape (d, n), reconstructing the original input format.
    """
    # Split the complex tensor into real and imaginary parts
    real_part = x.real
    imaginary_part = x.imag

    # Concatenate the real and imaginary parts along the feature dimension
    x = torch.cat((real_part, imaginary_part), dim=0)

    return x.to(device)


def complex_tensor(
    x: torch.Tensor,
    device: str = 'cpu',
) -> torch.Tensor:
    """Get the complex form of a tensor.

    Args:
        x : torch.Tensor
            The original tensor.
        device : str
            The device of the tensor. Default cpu.

    Returns:
        torch.Tensor
            The output tensor, which is the complex form of the original tensor.
    """
    device = x.device
    x = torch.stack((x, torch.zeros(x.shape).to(device)), dim=-1)
    return torch.view_as_complex(x).to(device)


def complex_gaussian_matrix(
    mean: float,
    std: float,
    size: tuple[int],
) -> torch.Tensor:
    """A method that returns a complex gaussian matrix in the torch.Tensor format.

    Args:
        mean : float
            The mean of the distribution.
        std : float
            The std of the distribution.
        size : tuple[int]
            The size of the matrix.

    Returns:
        torch.Tensor
            The complex gaussian matrix in the torch.Tensor format.
    """
    # Get the real and imaginary parts
    real_part = torch.normal(mean, std / 2, size=size)
    imag_part = torch.normal(mean, std / 2, size=size)

    # Stack real and imaginary parts along the last dimensioni
    complex_matrix = torch.stack((real_part, imag_part), dim=-1)

    return torch.view_as_complex(complex_matrix)


def awgn(
    sigma: float,
    size: torch.Size,
    device: str = 'cpu',
) -> torch.Tensor:
    """A function that returns a noise vector sampled by a complex gaussian of a specified sigma.

    Args:
        sigma : float
            The sigma (std) of a REAL awgn.
        size : torch.Size
            The size of the noise vector.
        device : str
            The device of the tensor. Default cpu.

    Returns:
        torch.Tensor
            The sempled noise vector.
    """
    # Get the complex sigma
    sigma = sigma / math.sqrt(2)

    # Get the real and imaginary parts
    r = torch.normal(mean=0, std=sigma, size=size)
    i = torch.normal(mean=0, std=sigma, size=size)

    return torch.view_as_complex(torch.stack((r, i), dim=-1)).to(device)


def snr(
    signal: torch.Tensor,
    sigma: float,
) -> float:
    """Return the Signal to Noise Ratio.

    Args:
        signal : torch.Tensor
            The signal vector.
        sigma : float
            The sigma of the noise.

    Return:
        float
            The Signal to Noise Ratio.
    """
    return (
        10 * torch.log10(torch.mean(torch.abs(signal) ** 2) / sigma**2).item()
    )


def sigma_given_snr(
    snr: float,
    signal: torch.Tensor,
) -> float:
    """Given a fixed value of SNR and signal, retrieve the correspoding value of sigma.

    Args:
        snr : float
            The Signal to Noise Ration in dB.
        signal : torch.Tensor
            The number of receiving antennas.
        cost : float
            The cost for the transmitter.

    Returns:
        float
            The corresponding sigma given snr and a signal.
    """
    snr_no_db = math.pow(10, snr / 10)
    signal_power = torch.mean(torch.abs(signal) ** 2)
    return math.sqrt(signal_power / snr_no_db)


def a_inv_times_b(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Perform in an efficient way the A^{-1}B.

    Args:
        a : torch.Tensor
            The original tensor to invert.
        b : torch.Tensor
            The original tensor to multiply.

    Returns:
        c : torch.Tensor
            The result of the multiplication.
    """
    try:
        c = torch.linalg.solve(a, b)
    except RuntimeError as e:
        if 'The input tensor A must have at least 2 dimensions' in str(e):
            if len(a.shape) == 2 and a.shape[0] == 1 and a.shape[-1]:
                c = (1 / a) @ b
            else:
                raise e
        else:
            raise e

    return c


def prewhiten(
    x_train: torch.Tensor,
    x_test: torch.Tensor = None,
    device: str = 'cpu',
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Prewhiten the training and test data using only training data statistics.

    Args:
        x_train : torch.Tensor
            The training torch.Tensor matrix.
        x_test : torch.Tensor
            The testing torch.Tensor matrix. Default None.
        device : str
            The device of the tensor. Default cpu.

    Returns:
        z_train : torch.Tensor
            Prewhitened training matrix.

        L : torch.Tensor
            The matrix resulting from the Cholesky decomposition.

        mean : torch.Tensor
            The mean of the x_train input.

        z_test : torch.Tensor
            Prewhitened test matrix.
    """
    # --- Prewhiten the training set ---
    C = torch.cov(x_train)  # Training set covariance
    try:
        L, _ = torch.linalg.cholesky_ex(C)  # Cholesky decomposition C = LL^H
    except RuntimeError as e:
        if 'The input tensor A must have at least 2 dimensions' in str(e):
            L = torch.sqrt(C).unsqueeze(0).unsqueeze(1)  # As a 1x1 tensor
        else:
            raise e
    mean = x_train.mean(axis=1)[:, None]
    z_train = x_train - mean  # Center the training set
    z_train = a_inv_times_b(L, z_train)  # Prewhitened training set

    if x_test is not None:
        z_test = x_test - mean  # Center the test set
        z_test = a_inv_times_b(L, x_test)  # Prewhitened training set
        return (
            z_train.to(device),
            L.to(device),
            mean.to(device),
            z_test.to(device),
        )

    return z_train.to(device), L.to(device), mean.to(device)


def remove_non_empty_dir(path: str) -> None:
    """
    Removes a non-empty directory given its path as a string.

    Parameters:
        path : str
            Path to the directory to remove.

    Raises:
        NotADirectoryError: If the path is not a directory.
        Exception: For any other error during deletion.
    """
    dir_path = Path(path)

    if not dir_path.exists():
        print(f"The path '{path}' does not exist.")
        return None

    if not dir_path.is_dir():
        raise NotADirectoryError(f"The path '{path}' is not a directory.")

    try:
        shutil.rmtree(dir_path)
        print(f'Successfully removed directory: {dir_path}')
    except Exception as e:
        raise Exception(f'Error while removing directory: {e}')

    return None


def mmse_svd_equalizer(
    channel_matrix: torch.tensor,
    snr_db: float = None,
):
    """
    Linear MMSE equalizer via SVD.

    Args:
        channel_matrix : torch.Tensor
            Channel matrix of shape (Nr, Nt), complex dtype.
        snr_db : float | None
            SNR in dB; if None, returns ZF equalizer.

    Returns:
        G, F : tuple[torch.tensor, torch.tensor]
    """
    # SVD: H = U @ diag(s) @ Vh
    U, s, Vh = torch.linalg.svd(channel_matrix)

    # Get antennas
    antennas_receiver, _ = channel_matrix.shape

    # Form diagonal Sigma and cast to complex dtype
    Sigma = torch.diag(s).to(channel_matrix.dtype)

    # Precoder = right singular vectors
    F = Vh.H

    if snr_db is not None:
        snr_linear = 10 ** (snr_db / 10)
        reg = (1.0 / snr_linear) * torch.eye(
            antennas_receiver, dtype=channel_matrix.dtype
        )

        # MMSE receiver
        inv_term = torch.inverse(Sigma.H @ Sigma + reg)
        G = inv_term @ (U @ Sigma).H
    else:
        # Zero-forcing: G = H^+ = V Σ⁻¹ U^H
        inv_Sigma = torch.diag(1.0 / s).to(channel_matrix.dtype)
        G = inv_Sigma @ U.H

    # Normalize F to unit Frobenius norm
    G *= torch.norm(F, p='fro')
    F /= torch.norm(F, p='fro')

    return G, F


# ================================================================
#
#                        Main Definition
#
# ================================================================


def main() -> None:
    """Some quality tests..."""

    # Variable definition
    mean: float = 0.0
    std: float = 1.0
    size: tuple[int] = (4, 4)

    n = 10
    d = 20
    x = torch.randn(n, d)

    print('Performing first test...', end='\t')
    complex_gaussian_matrix(mean=mean, std=std, size=size)
    print('[PASSED]')

    print()
    print('Performing second test...', end='\t')
    complex_tensor(x)
    print('[PASSED]')

    print()
    print('Performing third test...', end='\t')
    snr(x.real, std)
    print('[PASSED]')

    print()
    print('Performing fourth test...', end='\t')
    x_c = complex_compressed_tensor(x)
    print('[PASSED]')

    print()
    print('Performing fifth test...', end='\t')
    x_hat = decompress_complex_tensor(x_c)

    if not torch.all(torch.eq(x_hat[:, :d], x)):
        raise Exception(
            'The compression and decompression are not working as intended'
        )

    print('[PASSED]')

    print()
    print('Performing sixth test...', end='\t')
    prewhiten(x)
    print('[PASSED]')

    print()
    print('Performing seventh test...', end='\t')
    sigma = sigma_given_snr(snr=10, signal=x)
    assert sigma > 0, '[Error]: sigma is not positive.'
    print('[PASSED]')

    print()
    print('Performing eight test...', end='\t')
    awgn(sigma=sigma, size=x.shape)
    print('[PASSED]')

    return None


if __name__ == '__main__':
    main()
