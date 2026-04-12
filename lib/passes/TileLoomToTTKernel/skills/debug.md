# Debug Guide for LLM Code Review

## Purpose

Use this guide when reviewing Tenstorrent code for possible **hangs, deadlocks, stalls, or non-progress bugs** **without running the code**.

This guide is for **static code review only**. Focus on synchronization, queueing, dependency, and progress invariants. Do not speculate broadly. Do not propose fixes unless you can point to a likely broken invariant.

## Review Mode

When asked to inspect a possible hang:

1. Do **not** build or run anything unless explicitly requested.
2. Read the required repo guidance files first.
3. Check recent GitHub issues for already-known patterns.
4. Prioritize control flow, synchronization, queueing, and progress logic over performance tuning.
5. Return a report with concrete evidence, not guesses.

## Main Question

A hang review should first answer:

**Where can progress stop permanently?**

Everything in this guide is in service of that question.

## Review Order

### 1. Cross-Stage Progress Boundaries

Look first for places where one stage assumes another stage has completed:

* producer / consumer handoff
* reserve / commit
* enqueue / dequeue
* issue / completion
* send / receive
* write / signal
* launch / callback
* compile / execute boundary
* materialization / readiness boundary

These are the highest-value review targets because hangs usually come from broken progress contracts, not arithmetic mistakes.

### 2. Synchronization and Waiting Primitives

Prioritize review of:

* waits
* barriers
* semaphores
* condition variables
* polling loops
* futures / promises
* event completion checks
* stream synchronization
* queue draining
* mailbox protocols
* NoC send / receive coordination
* circular-buffer reserve / push / pop / wait semantics

Ask:

* What guarantees the waited-on state changes?
* Which actor changes it?
* Can that actor also block?
* Is there any path where the signal is skipped?

### 3. Early Return and Error Paths

For every acquire / reserve / wait / signal pair, inspect:

* early returns
* assertion failures
* exception paths
* cleanup paths
* cancellation paths
* timeout paths
* partial initialization failures

Common pattern: one side acquires or reserves, then exits through an error path without releasing or signaling the peer.

### 4. Shape, Tiling, and Partition Edge Cases

Focus on:

* zero-sized tiles
* partial tiles
* tail blocks
* uneven partitioning
* split-K or chunked reductions
* multicast fanout mismatch
* grid dimension mismatch
* head / group / chunk remapping math
* ceil-div on one side and floor-div on the other

Many non-progress bugs are really count mismatches in disguise.

### 5. State-Machine Completeness

Whenever code behaves like a protocol, reconstruct the states, for example:

* initialized
* reserved
* populated
* published
* consumed
* released

Then verify:

* every state transition is reachable
* no required transition can be skipped
* every terminal path releases downstream waiters
* retries cannot loop forever without outside progress

### 6. Compiler / Lowering Invariants

If reviewing compiler-facing code, inspect whether the hang may actually be caused by a bad lowering decision:

* illegal or inconsistent IR after transformation
* missing dependency edges
* incorrect decomposition or fusion
* wrong lifetime or ownership assumptions
* broken pass ordering
* missing synchronization ops in lowered IR

## High-Risk Patterns

Flag these aggressively.

### A. Unbounded Polling Loops

Examples:

* `while (!done) {}`
* retry loops with no proven producer
* polling state with no progress guarantee

Ask:

* What guarantees `done` changes?
* Who changes it?
* Can that actor itself block forever?

### B. Asymmetric Producer / Consumer Math

Examples:

* sender loops over `N`, receiver expects `M`
* producer includes tails, consumer drops them
* head/core/group mapping differs between stages
* one side uses different partition logic after refactoring

### C. Reserve / Wait Misuse

Check:

* reserve count matches publish / release count
* waited resources are always produced
* ownership is not duplicated
* nested reservation scopes cannot deadlock

### D. Hidden Dependency Through Shared State

Examples:

* mutable global scheduling state
* shared flags updated indirectly
* helper function mutates readiness state
* state written in one pass and assumed by another

### E. "Impossible" or "Unreachable" Branches

Any branch described as unreachable or guaranteed by prior passes deserves scrutiny. Those assumptions often fail first.

## What Not to Prioritize First

Do not start with:

* style cleanup
* naming improvements
* arithmetic simplification
* generic refactoring
* micro-optimizations
* broad performance advice

Those are secondary unless they directly explain a non-progress bug.

## Required Output Format

When reviewing suspected hang code, return the following:

### 1. Suspected Hang Sites

List the exact functions, blocks, or protocol boundaries most likely to stall.

### 2. Why Each Site Is Risky

For each site, explain the fragile or broken invariant, for example:

* missing signal
* mismatched counts
* asymmetric partitioning
* early return without release
* cyclic wait
* state transition that may never happen

### 3. Confidence Level

Use one of:

* **High**: a clear broken invariant is visible in code
* **Medium**: strong static evidence, but depends on caller assumptions
* **Low**: suspicious pattern, but runtime evidence would still be needed

### 4. Evidence

Quote or summarize only the minimal code needed to support the claim.

### 5. Next Inspection Target

State the next file or function to inspect. Do not end with a generic statement like "needs more debugging."

## Review Prompts

Use prompts like these:

* Review this code only for possible deadlock, stall, or non-progress bugs. Ignore style and optimization.
* Identify wait/signal asymmetries, producer/consumer count mismatches, and early-return paths that skip release.
* Reconstruct the protocol state machine and list transitions that can block forever.
* Check whether partitioning math is symmetric between sender and receiver.
* Assume the hang is caused by a broken invariant in synchronization or queueing. Find the most likely invariant.

## Checklist

Use this checklist during review:

* Does every wait have a guaranteed producer?
* Can any producer exit without signaling?
* Are loop bounds identical on both sides of a protocol?
* Are tail and partial tiles handled symmetrically?
* Are reserve, publish, consume, and release counts balanced?
* Can an error path strand a resource?
* Does lowering preserve the required dependency or barrier?
* Is there any cyclic wait across cores, queues, or stages?
* Is shared mutable state being used as implicit synchronization?
* Is this already a known issue in the tracker?