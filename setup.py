# Copyright 2023 Bloomberg Finance L.P.
# Distributed under the terms of the Apache 2.0 license.
# https://www.apache.org/licenses/LICENSE-2.0

from setuptools import find_namespace_packages, setup

setup(
    name="semantic_shrinkage",
    version="0.0.1",
    description="""Package for semantic similarity covariance shrinkage.""",
    author="Guillaume Becquin",
    author_email="gbecquin1@bloomberg.net",
    python_requires=">=3.8",
    include_package_data=True,
    packages=find_namespace_packages(
        include=[
            "semantic_shrinkage",
            "semantic_shrinkage.*",
        ]
    ),
    install_requires=[
        "torch>=1.12",
    ],
    zip_safe=False,
)
