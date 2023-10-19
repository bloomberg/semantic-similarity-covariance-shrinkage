# Copyright 2023 Bloomberg Finance L.P.
# Distributed under the terms of the Apache 2.0 license.
# https://www.apache.org/licenses/LICENSE-2.0

import pytest
import torch

from semantic_shrinkage.shrinkage import SemanticShrinkage
from semantic_shrinkage.shrinkage_factor import (
    compute_semantic_similarity_shrinkage_intensity,
)
from semantic_shrinkage.shrinkage_target import (
    compute_semantic_similarity_shrinkage_target,
)


class TestSemanticShrinkage:
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

    def test_create(self) -> None:
        # given
        shrinkage_target = compute_semantic_similarity_shrinkage_target(
            self.similarity_matrix, self.sample_covariance
        )
        shrinkage_intensity = compute_semantic_similarity_shrinkage_intensity(
            shrinkage_target, self.sample_covariance, self.returns
        )

        # when
        test_output = SemanticShrinkage(
            shrinkage_target, shrinkage_intensity, self.sample_covariance
        )

        # then
        assert isinstance(test_output, SemanticShrinkage)
        assert torch.all(test_output.sample_covariance == self.sample_covariance)
        assert test_output.shrinkage_intensity == shrinkage_intensity
        assert torch.all(test_output.shrinkage_target == shrinkage_target)

    def test_create_wrong_dimensions_fails(self) -> None:
        # given
        shrinkage_target = torch.rand(
            tuple(dim_size + 1 for dim_size in self.sample_covariance.shape)
        )
        shrinkage_intensity = 0.5

        # when & then
        with pytest.raises(ValueError):
            _ = SemanticShrinkage(
                shrinkage_target, shrinkage_intensity, self.sample_covariance
            )

    def test_create_shrinkage_intensity_out_of_bounds_fails(self) -> None:
        # given
        shrinkage_target = compute_semantic_similarity_shrinkage_target(
            self.similarity_matrix, self.sample_covariance
        )
        shrinkage_intensity = 1.1

        # when & then
        with pytest.raises(ValueError):
            _ = SemanticShrinkage(
                shrinkage_target, shrinkage_intensity, self.sample_covariance
            )

    def test_create_from_sample_covariance_matrix(self) -> None:
        # when
        test_output = SemanticShrinkage.from_returns_and_sample_covariance(
            self.returns, self.sample_covariance, self.similarity_matrix
        )

        # then
        assert isinstance(test_output, SemanticShrinkage)
        assert torch.all(test_output.sample_covariance == self.sample_covariance)
        assert isinstance(test_output.shrinkage_intensity, float)
        assert test_output.shrinkage_intensity == pytest.approx(0.4655, 1e-3)
        assert isinstance(test_output.shrinkage_target, torch.Tensor)
        torch.testing.assert_close(
            test_output.shrinkage_target,
            torch.tensor(
                [
                    [0.1569, 0.0021, 0.0529, 0.0959],
                    [0.0021, 0.1629, -0.0783, 0.0281],
                    [0.0529, -0.0783, 0.1909, 0.0425],
                    [0.0959, 0.0281, 0.0425, 0.2327],
                ]
            ),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_create_from_returns(self) -> None:
        # when
        test_output = SemanticShrinkage.from_returns(
            self.returns, self.similarity_matrix
        )

        # then
        assert isinstance(test_output, SemanticShrinkage)
        assert torch.all(test_output.sample_covariance == self.sample_covariance)
        assert isinstance(test_output.shrinkage_intensity, float)
        assert test_output.shrinkage_intensity == pytest.approx(0.4655, 1e-3)
        assert isinstance(test_output.shrinkage_target, torch.Tensor)
        torch.testing.assert_close(
            test_output.shrinkage_target,
            torch.tensor(
                [
                    [0.1569, 0.0021, 0.0529, 0.0959],
                    [0.0021, 0.1629, -0.0783, 0.0281],
                    [0.0529, -0.0783, 0.1909, 0.0425],
                    [0.0959, 0.0281, 0.0425, 0.2327],
                ]
            ),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_create_get_shrunk_covariance_matrix(self) -> None:
        # given
        semantic_shrinkage = SemanticShrinkage.from_returns(
            self.returns, self.similarity_matrix
        )

        # when
        test_output = semantic_shrinkage.get_shrunk_covariance()

        # then
        assert isinstance(test_output, torch.Tensor)
        torch.testing.assert_close(
            test_output,
            torch.tensor(
                [
                    [0.1569, 0.0326, 0.0530, 0.0428],
                    [0.0326, 0.1629, -0.0606, -0.0043],
                    [0.0530, -0.0606, 0.1909, 0.0975],
                    [0.0428, -0.0043, 0.0975, 0.2327],
                ]
            ),
            rtol=1e-4,
            atol=1e-4,
        )
