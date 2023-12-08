# Semantic Similarity Covariance Matrix Shrinkage

This repository contains the supporting code for [Semantic Similarity Covariance Matrix Shrinkage](https://aclanthology.org/2023.findings-emnlp.668/), published in [Findings of EMNLP 2023](https://2023.emnlp.org).
It implements the methods for shrinking covariance matrices using a cosine similarity target.

## Menu

- [Installation](#installation)
- [Quick start](#quick-start)
- [Contributions](#contributions)
- [License](#license)
- [Code of Conduct](#code-of-conduct)
- [Security Vulnerability Reporting](#security-vulnerability-reporting)


## Installation

This project requires Python 3.8 or greater. Clone the repository and install the module:
```
python3.8 -m pip install .
```

## Quick Start

The library requires a cosine similarity matrix that can be generated from normalized embeddings as an input. Assuming a set of `k` `embeddings` of dimension `p` stored in a PyTorch `[k,p]` tensor, the similarity matrix can be built using:

```python
import torch

normalized_embeddings = torch.nn.functional.normalize(embeddings)
similarity_matrix = normalized_embeddings @ normalized_embeddings.t()
```

Assuming the random variable observations (e.g., stock price returns) are available as a `[N,p]` tensor called `returns`, the shrunk covariance matrix can be computed directly using:
```python
from semantic_shrinkage import SemanticShrinkage

shrunk_covariance_matrix = SemanticShrinkage.from_returns(
    returns, similarity_matrix
).get_shrunk_covariance()
```

## Contributions

We :heart: contributions.

Have you had a good experience with this project? Why not share some love and contribute code, or just let us know about any issues you had with it?

We welcome issue reports [here](../../issues); be sure to choose the proper issue template for your issue, so that we can be sure you're providing the necessary information.

Before sending a [Pull Request](../../pulls), please make sure you read our
[Contribution Guidelines](https://github.com/bloomberg/.github/blob/master/CONTRIBUTING.md).

## License

Please read the [LICENSE](LICENSE) file.

## Code of Conduct

This project has adopted a [Code of Conduct](https://github.com/bloomberg/.github/blob/master/CODE_OF_CONDUCT.md).
If you have any concerns about the Code, or behavior which you have experienced in the project, please
contact us at opensource@bloomberg.net.

## Security Vulnerability Reporting

Please refer to the project [Security Policy](https://github.com/bloomberg/semantic-similarity-covariance-shrinkage/security/policy).
