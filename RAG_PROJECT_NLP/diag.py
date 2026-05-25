"""
test_faiss.py
Diagnose faiss installation inside Modal
"""

import modal

app = modal.App("test-faiss")

# Try different faiss versions
test_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(["numpy==1.26.4"])
    .pip_install(["faiss-cpu==1.7.4"])
)

@app.function(image=test_image, timeout=300)
def test_faiss():
    import sys
    import subprocess

    print(f"Python: {sys.version}")

    # Show what is installed
    result = subprocess.run(
        ["pip", "show", "faiss-cpu"],
        capture_output=True,
        text=True
    )
    print(f"\nfaiss-cpu info:\n{result.stdout}")

    # Show all installed packages
    result2 = subprocess.run(
        ["pip", "list"],
        capture_output=True,
        text=True
    )
    print(f"\nAll packages:\n{result2.stdout}")

    # Try importing
    try:
        import faiss
        print(f"\n✅ faiss imported successfully")
        print(f"faiss file: {faiss.__file__}")
        print(f"\nAll faiss attributes:")
        for attr in sorted(dir(faiss)):
            print(f"  {attr}")
    except Exception as e:
        print(f"\n❌ faiss import failed: {e}")

    # Try numpy
    try:
        import numpy as np
        print(f"\n✅ numpy version: {np.__version__}")
    except Exception as e:
        print(f"\n❌ numpy failed: {e}")

    return "done"

@app.local_entrypoint()
def main():
    result = test_faiss.remote()
    print(result)