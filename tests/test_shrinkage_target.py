# Copyright 2023 Bloomberg Finance L.P.
# Distributed under the terms of the Apache 2.0 license.
# https://www.apache.org/licenses/LICENSE-2.0

import torch

from semantic_shrinkage.shrinkage_target import (
    compute_semantic_similarity_shrinkage_target,
    make_positive_semi_definite,
    scale_with_variance,
)


class TestShrinkageTarget:
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

    def test_scale_with_variance(self) -> None:
        # given & when
        test_output = scale_with_variance(
            self.similarity_matrix, self.sample_covariance
        )

        # then
        assert isinstance(test_output, torch.Tensor)
        torch.testing.assert_close(
            test_output,
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

    def test_make_positive_semi_definite(self) -> None:
        # given
        scaled_similarity = torch.tensor(
            [
                [0.1569, 0.1121, 0.0529, 0.0959],
                [0.1121, 0.1629, -0.0783, 0.0281],
                [0.0529, -0.0783, 0.1909, 0.0425],
                [0.0959, 0.0281, 0.0425, 0.2327],
            ]
        )
        eigenvalues, _ = torch.linalg.eigh(scaled_similarity)
        assert not (torch.all(eigenvalues > 0))

        # when
        test_output = make_positive_semi_definite(scaled_similarity)

        # then
        assert isinstance(test_output, torch.Tensor)
        torch.testing.assert_close(
            test_output,
            torch.tensor(
                [
                    [0.1575, 0.1121, 0.0529, 0.0959],
                    [0.1121, 0.1635, -0.0783, 0.0281],
                    [0.0529, -0.0783, 0.1915, 0.0425],
                    [0.0959, 0.0281, 0.0425, 0.2333],
                ]
            ),
            rtol=1e-4,
            atol=1e-4,
        )
        output_eigenvalues, _ = torch.linalg.eigh(test_output)
        assert torch.all(output_eigenvalues > 0)

    def test_compute_semantic_similarity_target(self) -> None:
        # given & when
        test_output = compute_semantic_similarity_shrinkage_target(
            self.similarity_matrix, self.sample_covariance
        )

        # then
        assert isinstance(test_output, torch.Tensor)
        torch.testing.assert_close(
            test_output,
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
        output_eigenvalues, _ = torch.linalg.eigh(test_output)
        assert torch.all(output_eigenvalues > 0)
