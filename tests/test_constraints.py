import sys

def test_no_heavy_dependencies():
    """
    Ensure the CLI entry point does not accidentally import heavy ML libraries.
    The primary value proposition of modelinfo is sub-100ms startup times via zero-dependencies.
    """
    # Import the cli directly to populate sys.modules
    import modelinfo.cli  # noqa: F401
    
    forbidden_modules = ["torch", "transformers", "numpy", "safetensors"]
    for mod in forbidden_modules:
        assert mod not in sys.modules, f"Regression: {mod} was imported!"
