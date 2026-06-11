"""FastAPI backend for the APR Research dashboard.

Serves typed ``/api/*`` endpoints over the frozen ``results/`` artifacts,
reusing ``src.evaluation.metrics`` so every number equals the canonical
``results/`` files and the README. Categorization for the A/B CV datasets is
derived at the base upset rate (never 0.5), matching ``DisagreementAnalyzer``.
"""

from __future__ import annotations
