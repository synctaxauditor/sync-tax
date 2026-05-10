# Understanding the Synchronization Tax in GPU Scale-Up Domains
Code for the paper, _Understanding the Synchronization Tax in GPU Scale-Up Domains_. This is meant to provide a general overview of our implementation. We will clean up the code for the artifact evaluation. We are still working on data upload due to each workload trace file being several GBs.

- `diagnostics`: contains the code for Sections 3 and 4 of the paper, _i.e.,_ critical-path analysis algorithm, blame attribution, etc.
- `analytical`: contains the code for Sections 5 and 6 of the paper, _i.e.,_ analytical model and validation for the synchronization tax (via Generalized Extreme Value Distribution) and bandwidth scaling model
