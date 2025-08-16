from functools import wraps


def apply_patches():
    """Apply all patches to fix Streamlit and PyTorch compatibility issues."""
    patch_streamlit_watcher()


def patch_streamlit_watcher():
    """
    Patch Streamlit's local_sources_watcher to handle PyTorch's custom classes.

    The issue occurs when Streamlit tries to examine torch.classes.__path__._path,
    which triggers PyTorch's custom __getattr__ mechanism and causes an error.
    """
    try:
        import streamlit.watcher.local_sources_watcher as watcher

        # Store the original get_module_paths function
        original_get_module_paths = watcher.get_module_paths

        @wraps(original_get_module_paths)
        def patched_get_module_paths(module):
            """
            Patched version of get_module_paths that handles PyTorch's custom classes.
            """
            # Skip processing for torch.classes module
            if hasattr(module, "__name__") and module.__name__ == "torch.classes":
                return []

            # For all other modules, use the original function
            return original_get_module_paths(module)

        # Replace the original function with our patched version
        watcher.get_module_paths = patched_get_module_paths

        print("✅ Successfully applied patch for Streamlit and PyTorch compatibility")
    except ImportError:
        print("⚠️ Could not patch Streamlit watcher (streamlit may not be installed)")
    except Exception as e:
        print(f"⚠️ Failed to apply Streamlit patch: {e}")


# Apply patches when this module is imported
apply_patches()
