# Copyright 2023 Bloomberg Finance L.P.
# Distributed under the terms of the Apache 2.0 license.
# https://www.apache.org/licenses/LICENSE-2.0

import torch


def scale_with_variance(
    similarity_matrix: torch.Tensor, sample_covariance: torch.Tensor
) -> torch.Tensor:
    r"""Scale the similarity matrix by the variance.

    This is equivalent to converting a correlation to a covariance matrix. This assumes
    the provided similarity matrix is morphologically similar to a correlation (symmetric,
    diagonal entries equal to 1,  off-diagonal between -1 and 1).

    The conversion performed assuming the semantic similarity is a correlation matrix to be
    converted to a covariance and follows the definition:
        .. math::
        corr(x,y) := \frac{cov(x,y)}{\sqrt{var(x)var(y)}}

    Args:
        similarity_matrix (torch.Tensor): Semantic similarity matrix
        sample_covariance (torch.Tensor): Sample covariance matrix

    Returns:
        torch.Tensor: Similarity matrix scaled by variances
    """
    vars = sample_covariance.diagonal().sqrt()
    scaling_matrix = vars.unsqueeze(1) @ vars.unsqueeze(0)
    return similarity_matrix * scaling_matrix


def make_positive_semi_definite(
    input_matrix: torch.Tensor, epsilon: float = 1e-8
) -> torch.Tensor:
    """Perturb a target to make it positive semi-definite.

    Computes the spectral decomposition of the input matrix and
    shifts them by the minimum amount for them to be all positive.

    Args:
        input (torch.Tensor): Matrix to check for (and correct) positive semi-definiteness
        epsilon (float, defaults to 1e-8): Minimum value for all eigenvalues

    Returns:
        torch.Tensor: Minimally perturbed, positive semi-definite version of the input matrix
    """
    output = input_matrix.double()
    eigenvalues, eigenvectors = torch.linalg.eigh(input_matrix)
    if torch.all(eigenvalues > 0):
        return input_matrix
    eigenvalues += eigenvalues.min().abs() + epsilon
    output = eigenvectors @ eigenvalues.diag() @ eigenvectors.t()
    return output


def compute_semantic_similarity_shrinkage_target(
    similarity_matrix: torch.Tensor,
    sample_covariance: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Compute a shrinkage target from the similarity matrix.

    Compute a covariance-equivalent to the similarity-based correlation matrix
    and correct it for positive semi-definiteness if needed.

    Args:
        similarity_matrix (torch.Tensor): Semantic similarity matrix
        sample_covariance (torch.Tensor): Sample covariance matrix
        epsilon (float, defaults to 1e-8): Minimum value for all eigenvalues

    Returns:
        torch.Tensor: Semantic similarity shrinkage target
    """
    target = scale_with_variance(similarity_matrix, sample_covariance)
    target = make_positive_semi_definite(target, epsilon)
    return target
