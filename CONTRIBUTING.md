# Contributing to NeuroBridge

We welcome contributions from the biocomputing community.

## Quick Start

```bash
git clone https://github.com/Luckyguybiz/neurobridge-api.git
cd neurobridge-api
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install pytest httpx
python main.py  # API at localhost:8847
```

## Running Tests

```bash
python -m pytest tests/ -v
```

All 119 tests must pass before submitting a PR.

## Adding a New Analysis Module

1. Create `analysis/your_module.py`
2. Import `SpikeData` from `.loader`
3. Write functions that accept `SpikeData` and return `dict`
4. Add docstring with scientific basis
5. Wire endpoint in `main.py`
6. Add test in `tests/test_new_modules.py`

### Module Template

```python
"""Module description — scientific basis.

Reference: Author et al., Journal, Year.
"""
import numpy as np
from .loader import SpikeData

def your_analysis(data: SpikeData, param: float = 1.0) -> dict:
    """One-line description.

    Args:
        data: SpikeData from organoid recording.
        param: Description of parameter.

    Returns:
        Dict with analysis results.
    """
    # Your analysis here
    return {"result": value}
```

## Code Style

- Type hints on all public functions
- Docstrings (Google style)
- Return dicts (JSON-serializable)
- Use `_sanitize()` wrapper in main.py endpoints for numpy types

## Reporting Issues

Open an issue at https://github.com/Luckyguybiz/neurobridge-api/issues with:
- What you expected
- What happened
- Steps to reproduce
- Python version and OS

## License

MIT. By contributing, you agree your code will be released under MIT.
