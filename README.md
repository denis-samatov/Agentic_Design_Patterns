# Agentic Design Patterns
Agentic Design Patterns: A Hands-On Guide to Building Intelligent Systems by Antonio Gulli
## Fork scope and local example

This is a fork of [AnkunHuang/Agentic_Design_Patterns](https://github.com/AnkunHuang/Agentic_Design_Patterns).
The book and original chapter examples remain upstream work. This fork includes
changes to the resource-aware routing example: JSON/error handling and offline
regression tests. It is educational material, not a production agent framework.

The resource-aware example has a separate dependency input in
`requirements-example.in`. Install it in an isolated environment and run
`python -m pytest notebooks/test_resource_aware_optimization.py notebooks/test_router_contract.py`.
Importing that module does not construct an API client or read a `.env` file;
actual model/search calls require your own credentials. Tests make no API calls.


For the repaired runnable example only, install the hash-pinned profile with
`pip install --require-hashes -r requirements-example.lock` and run
`python -m pytest notebooks -q`. The other book examples are not qualified by
that environment or by the offline router tests.
