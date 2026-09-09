# Agent design notes

The agent retries up to three times before giving up, and every validation
failure is logged with the raw model output, the error and the attempt number.
Retries feed the exact validation error back to the model, which is why the
second attempt usually succeeds instead of repeating the same mistake.

# Routing and budgets

Routing picks a small model for simple tasks and a frontier model for hard
ones. Token budgets are enforced before the call is made, not discovered
afterwards. Escalation is one way: a cheap model that hedges is retried once on
a bigger model, and both calls are written to the cost ledger.

# Memory

Memory has two tiers. A short-term buffer holds recent turns and summarises its
oldest half when it overflows. A long-term vector store keeps durable facts per
user and survives a restart, because it is written to disk rather than held in
process memory.

# Failure handling

Events that exhaust their retries land in a dead-letter queue with the full
payload and the failure history, so they can be replayed once the underlying
bug is fixed rather than silently lost.
