## libraries
import logging
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

## loads, validates, and returns a notebook cache
def load_notebook_cache(
    cache_path: Path,
    metadata: Mapping[str, Any],
    required_keys: Sequence[str],
    force_recompute: bool = False,
    ) -> dict[str, Any] | None:

    """
    Desc: Load a trusted local producer cache when its settings and outputs match.
    Args:
        cache_path: Path to a locally generated pickle file. Only trusted files
            may be loaded because pickle can execute code.
        metadata: Expected settings, including data size and model names.
        required_keys: Result entries needed by the notebook's analysis cells.
        force_recompute: Ignore an existing cache and request fresh computation.
    Returns:
        Compatible payload, or None when recomputation is needed.
    """

    ## check if recomputation is forced
    if force_recompute:
        print(f"Forced recomputation for {cache_path}")
        return None

    ## attempt to load the cache
    try:
        with cache_path.open(mode = "rb") as cache_file:
            payload = pickle.load(file = cache_file)
    except FileNotFoundError:
        print(f"Missing cache at {cache_path}")
        return None
    except (OSError, EOFError, pickle.UnpicklingError, ImportError, AttributeError, ValueError) as error:
        logging.warning("Unreadable cache at %s: %s", cache_path, error)
        return None

    ## validate the cache contents
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("metadata"), dict)
        or any(payload["metadata"].get(key) != value for key, value in metadata.items())
        or any(key not in payload or payload[key] is None for key in required_keys)
    ):
        print(f"Incompatible cache at {cache_path}")
        return None

    ## return the validated cache
    print(f"Loaded results from {cache_path}")
    return payload