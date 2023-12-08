# Copyright 2023 Bloomberg Finance L.P.
# Distributed under the terms of the Apache 2.0 license.
# https://www.apache.org/licenses/LICENSE-2.0

import torch

from semantic_shrinkage.shrinkage_factor import (
    compute_semantic_similarity_shrinkage_intensity,
)
from semantic_shrinkage.shrinkage_target import (
    compute_semantic_similarity_shrinkage_target,
)


class SemanticShrinkage:
    """Methods to the shrunk covariance matrix with a semantic similarity target."""

    def __init__(
        self,
        shrinkage_target: torch.Tensor,
        shrinkage_intensity: float,
        sample_covariance: torch.Tensor,
    ) -> None:
        """Create a new `SemanticShrinkage`.

        Args:
            shrinkage_target (torch.Tensor): Semantic similarity shrinkage target of shape [p, p]
            shrinkage_intensity (float): Shrinkage intensity/factor
            sample_covariance (torch.Tensor): Sample covariance matrix [p, p]
        """
        if shrinkage_intensity > 1 or shrinkage_intensity < 0:
            raise ValueError(
                f"`shrinkage_intensity` must be between 0 and 1, got {shrinkage_intensity}"
            )
        if shrinkage_target.shape != sample_covariance.shape:
            raise ValueError(
                f"""Expected `shrinkage_target` and `sample_covariance` to be of the same size, got
                - shrinkage_intensity: {shrinkage_target.shape}
                - sample_covariance:   {sample_covariance.shape},
                """
            )
        self.shrinkage_target = shrinkage_target
        self.shrinkage_intensity = shrinkage_intensity
        self.sample_covariance = sample_covariance

    @classmethod
    def from_returns_and_sample_covariance(
        cls,
        returns: torch.Tensor,
        sample_covariance: torch.Tensor,
        similarity_matrix: torch.Tensor,
    ) -> "SemanticShrinkage":
        """Create a new `SemanticShrinkage` from returns and sample_covariance tensors.

        The shrinkage target and intensities are calculated using the similarity matrix provided.

        Args:
            returns (torch.Tensor): Matrix of stock price returns of shape [N, p]
            sample_covariance (torch.Tensor): Sample covariance matrix of shape [p, p]
            similarity_matrix (torch.Tensor): Semantic similarity matrix [p, p]
        """
        shrinkage_target = compute_semantic_similarity_shrinkage_target(
            similarity_matrix, sample_covariance
        )
        shrinkage_intensity = compute_semantic_similarity_shrinkage_intensity(
            shrinkage_target, sample_covariance, returns
        )
        return cls(shrinkage_target, shrinkage_intensity, sample_covariance)

    @classmethod
    def from_returns(
        cls,
        returns: torch.Tensor,
        similarity_matrix: torch.Tensor,
    ) -> "SemanticShrinkage":
        """Create a new `SemanticShrinkage` from a returns tensors.

        The shrinkage target and intensities are calculated using the similarity matrix provided.

        Args:
            returns (torch.Tensor): Matrix of stock price returns of shape [N, p]
            similarity_matrix (torch.Tensor): Semantic similarity matrix [p, p]
        """
        sample_covariance = torch.cov(returns.t())
        return cls.from_returns_and_sample_covariance(
            returns, sample_covariance, similarity_matrix
        )

    def get_shrunk_covariance(self) -> torch.Tensor:
        """Get the shrunk covariance matrix.

        Returns:
            torch.Tensor: shrunk covariance matrix using a target derived from the similarity matrix.
        """
        return (
            self.sample_covariance * (1 - self.shrinkage_intensity)
            + self.shrinkage_intensity * self.shrinkage_target
        )
