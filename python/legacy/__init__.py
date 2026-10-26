"""Local legacy reference implementations used by the Python oracle.

Arithmetic and evaluation implementations are preserved. Imports use relative
package names, and the float bridge resolves only this local tree; importing
both reference implementations never installs generic model/core/evaluate modules.
Legacy trainer tape helpers are retained for evaluation compatibility only;
commit-then-challenge run functions exclusively use random_tape.CounterRng.
"""
