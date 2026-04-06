"""Environment variable loader using varlock and 1Password."""

import os
import subprocess
from pathlib import Path
from typing import Optional


def load_env_from_varlock(schema_path: str = "env.schema") -> dict:
    """Load environment variables using varlock.

    Args:
        schema_path: Path to the env.schema file

    Returns:
        Dict of resolved environment variables
    """
    schema = Path(schema_path)
    if not schema.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    # Use varlock to resolve all variables
    try:
        result = subprocess.run(
            ["varlock", "load", "--schema", str(schema)],
            capture_output=True,
            text=True,
            check=True,
        )

        # Parse the output (VAR=value format)
        env_vars = {}
        for line in result.stdout.strip().split("\n"):
            if "=" in line:
                key, value = line.split("=", 1)
                env_vars[key] = value

        return env_vars
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load env from varlock: {e.stderr}")


def load_single_env(key: str, schema_path: str = "env.schema") -> Optional[str]:
    """Load a single environment variable using varlock.

    Args:
        key: Variable name to load
        schema_path: Path to the env.schema file

    Returns:
        Resolved value or None if not found
    """
    schema = Path(schema_path)
    if not schema.exists():
        return None

    try:
        result = subprocess.run(
            ["varlock", "printenv", key, "--schema", str(schema)],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable with fallback to varlock.

    Args:
        key: Variable name
        default: Fallback value if not found

    Returns:
        Environment variable value
    """
    # First check os.environ
    value = os.environ.get(key)
    if value:
        return value

    # Try varlock if 1Password schema exists
    try:
        return load_single_env(key)
    except Exception:
        pass

    return default


def get_required_env(key: str) -> str:
    """Get required environment variable from varlock or raise.

    Args:
        key: Variable name

    Returns:
        Resolved value

    Raises:
        RuntimeError: If variable not found
    """
    value = get_env(key)
    if value is None:
        raise RuntimeError(f"Required environment variable not found: {key}")
    return value
