# Copyright 2023 Bloomberg Finance L.P.
# Distributed under the terms of the Apache 2.0 license.
# https://www.apache.org/licenses/LICENSE-2.0

import pytest
import torch

from semantic_shrinkage.shrinkage_factor import (
    _compute_pi_mat,
    _compute_rho_hat,
    compute_semantic_similarity_shrinkage_intensity,
)
from semantic_shrinkage.shrinkage_target import (
    compute_semantic_similarity_shrinkage_target,
)


class TestShrinkageFactor:
    def setup_class(self) -> None:
        embeddings = torch.tensor(
            [
                [0.1, -0.2, 0.3, 0.5, 0.6, 0.8, -0.9],
                [0.5, 0.2, 0.3, 0.5, -0.2, 0.3, 0.5],
                [0.1, 0.6, 0.3, -0.5, 0.7, -0.3, -0.8],
                [0.1, 0.2, -0.3, 0.5, 0.8, 0.2, 0.0],
            ]
        )

        normalized_embeddings = torch.nn.functional.normalize(embeddings)
        similarity_matrix = normalized_embeddings @ normalized_embeddings.t()
        assert similarity_matrix.shape == torch.Size([4, 4])
        self.similarity_matrix = similarity_matrix

        returns = torch.tensor(
            [
                [0.5, 0.3, 0.3, 0.5],
                [0.5, 0.2, 0.3, 0.5],
                [0.3, 0.9, -0.5, -0.5],
                [0.1, 0.2, -0.3, 0.5],
                [-0.1, -0.7, 0.3, 0.5],
                [0.1, 0.2, -0.8, -0.5],
                [-0.8, 0.2, -0.3, 0.7],
                [0.1, 0.2, -0.4, 0.5],
                [-0.5, -0.2, -0.8, -0.5],
                [-0.2, 0.2, 0.3, 0.5],
                [0.1, 0.6, -0.4, 0.5],
            ]
        )
        sample_covariance = torch.cov(returns.t())
        assert sample_covariance.shape == torch.Size([4, 4])
        self.returns = returns
        self.sample_covariance = sample_covariance

    def test_compute_pi_mat(self) -> None:
        # given
        N = 11
        demeaned_returns = self.returns - self.returns.mean(dim=0)

        # when
        test_output = _compute_pi_mat(self.sample_covariance, demeaned_returns, N)

        # then
        assert isinstance(test_output, torch.Tensor)
        torch.testing.assert_close(
            test_output,
            torch.tensor(
                [
                    [0.0319, 0.0052, 0.0195, 0.0334],
                    [0.0052, 0.0584, 0.0262, 0.0378],
                    [0.0195, 0.0262, 0.0111, 0.0251],
                    [0.0334, 0.0378, 0.0251, 0.0366],
                ]
            ),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_compute_rho_hat(self) -> None:
        # given
        p = 4
        N = 11
        demeaned_returns = self.returns - self.returns.mean(dim=0)
        pi_mat = _compute_pi_mat(self.sample_covariance, demeaned_returns, N)
        shrinkage_target = compute_semantic_similarity_shrinkage_target(
            self.similarity_matrix, self.sample_covariance
        )

        # when
        test_output = _compute_rho_hat(
            shrinkage_target, self.sample_covariance, demeaned_returns, pi_mat, N, p
        )

        # then
        assert isinstance(test_output, float)
        assert test_output == pytest.approx(0.1407, 1e-3)

    def test_compute_semantic_similarity_shrinkage_intensity(self) -> None:
        # given
        shrinkage_target = compute_semantic_similarity_shrinkage_target(
            self.similarity_matrix, self.sample_covariance
        )

        # when
        test_output = compute_semantic_similarity_shrinkage_intensity(
            shrinkage_target, self.sample_covariance, self.returns
        )

        # then
        assert isinstance(test_output, float)
        assert test_output == pytest.approx(0.4655, 1e-3)
