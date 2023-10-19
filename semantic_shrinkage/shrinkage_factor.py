# Copyright 2023 Bloomberg Finance L.P.
# Distributed under the terms of the Apache 2.0 license.
# https://www.apache.org/licenses/LICENSE-2.0

# The computation of the shrinkage intensity is based upon https://github.com/pald22/covShrinkage
# Shared under MIT license by Patrick Ledoit.
# The calculations have been modified and adapted to constant target shrinkage, shared under
# the same license as the rest of the files in this repository
#
# Copyright 2022 Patrick Ledoit
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch


def compute_semantic_similarity_shrinkage_intensity(
    shrinkage_target: torch.Tensor,
    sample_covariance: torch.Tensor,
    returns: torch.Tensor,
) -> float:
    """Compute the shrinkage factor from the similarity matrix.

    Estimators for the pi hat, rho hat and gamma hat components from
    "Improved Estimation of the Covariance Matrix of Stock Returns With an Application to Portfolio Selection",
    Ledoit et. al, 2001 (equation 4)

    Args:
        shrinkage_target (torch.Tensor): Semantic similarity shrinkage target of shape [p, p]
        sample_covariance (torch.Tensor): Sample covariance matrix of shape [p, p]
        returns (torch.Tensor): Matrix of stock price returns of shape [N, p]

    Returns:
        float: Shrinkage intensity
    """
    N, p = returns.shape
    demeaned_returns = returns - returns.mean(dim=0)

    # pi
    pi_mat = _compute_pi_mat(sample_covariance, demeaned_returns, N)
    pi_hat = pi_mat.sum().item()

    # gamma
    gamma_hat = (sample_covariance - shrinkage_target).norm("fro") ** 2

    # rho
    rho_hat = _compute_rho_hat(
        shrinkage_target, sample_covariance, demeaned_returns, pi_mat, N, p
    )

    shrinkage_intensity: float = (pi_hat - rho_hat) / gamma_hat.item() / N
    return max(0, min(1, shrinkage_intensity))


def _compute_pi_mat(
    sample_covariance: torch.Tensor, demeaned_returns: torch.Tensor, N: int
) -> torch.Tensor:
    """Compute the pi mat (variance of the sample covariance matrix).

    Args:
        sample_covariance (torch.Tensor): Sample covariance matrix of shape [p, p]
        demeaned_returns (torch.Tensor): Matrix of demeaned stock price returns of shape [N, p]
        N (int): number of observations (length of returns matrix)

    Returns:
        torch.Tensor: Variance of the covariance
    """
    y = demeaned_returns**2
    square_sample_covariance = y.T @ y
    pi_mat = square_sample_covariance / N - sample_covariance**2
    return pi_mat


def _compute_rho_hat(
    shrinkage_target: torch.Tensor,
    sample_covariance: torch.Tensor,
    demeaned_returns: torch.Tensor,
    pi_mat: torch.Tensor,
    N: int,
    p: int,
) -> float:
    """Compute the rho hat component of the shrinkage intensity.

    Args:
        shrinkage_target (torch.Tensor): Semantic similarity shrinkage target of shape [p, p]
        sample_covariance (torch.Tensor): Sample covariance matrix of shape [p, p]
        demeaned_returns (torch.Tensor): Matrix of demeaned stock price returns of shape [N, p]
        N (int): number of observations (length of returns matrix)
        p (int): number of variables (width of returns matrix)

    Returns:
        float: rho hat factor
    """
    rho_diag = sum(torch.diag(pi_mat))
    variances = torch.diag(sample_covariance)
    standard_deviations = variances.sqrt()
    term1 = ((demeaned_returns**3).t() @ demeaned_returns) / N
    term2 = variances.repeat(p, 1).t() * sample_covariance
    theta_mat = term1 - term2
    theta_mat[torch.eye(p) == 1] = torch.zeros(
        p, dtype=theta_mat.dtype, device=theta_mat.device
    )
    rho_off = (
        shrinkage_target
        * ((1 / standard_deviations).unsqueeze(1) @ standard_deviations.unsqueeze(0))
        * theta_mat
    ).sum()
    rho_hat: float = (rho_diag + rho_off).item()
    return rho_hat
