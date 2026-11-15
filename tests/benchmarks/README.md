# Provider benchmark

The benchmark must use the same questions, retrieved evidence, response
language, and evaluator for each provider. Live execution is intentionally
separate from the ordinary pytest suite because it requires provider access.

Results must be written as machine-readable JSON and must include sample size,
provider/model, latency, failure rate, structured-output validity, and the
evaluation criteria used. No result is considered acceptance evidence until it
has been produced by a real run.
