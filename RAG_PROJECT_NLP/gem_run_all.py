"""
gem_run_all.py

Master script — runs the complete GEM RAG pipeline in order.
All heavy steps run on Modal. Streamlit runs locally.

Usage:
    python gem_run_all.py              # full pipeline
    python gem_run_all.py --from=4    # start from step 4 (embed)
    python gem_run_all.py --step=6    # run only step 6 (retriever test)
    python gem_run_all.py --deploy    # deploy API endpoint
    python gem_run_all.py --ui        # launch Streamlit UI
    python gem_run_all.py --eval      # run evaluator
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path

# ─────────────────────────────────────────────
# Pipeline Steps
# ─────────────────────────────────────────────
STEPS = [
    {
        "number":  1,
        "name":    "Scraper",
        "file":    "gem_scraper.py",
        "command": "modal run gem_scraper.py",
        "desc":    "Scrape gem.eg (Arabic + English) via Playwright",
        "output":  "all_documents.json",
        "time":    "~15 min",
    },
    {
        "number":  2,
        "name":    "Cleaner",
        "file":    "gem_cleaner.py",
        "command": "modal run gem_cleaner.py",
        "desc":    "Clean and normalise scraped text",
        "output":  "cleaned_documents.json",
        "time":    "~2 min",
    },
    {
        "number":  3,
        "name":    "Chunker",
        "file":    "gem_chunker.py",
        "command": "modal run gem_chunker.py",
        "desc":    "Split documents into token-sized chunks",
        "output":  "all_chunks.json",
        "time":    "~2 min",
    },
    {
        "number":  4,
        "name":    "Embedder",
        "file":    "gem_embedder.py",
        "command": "modal run gem_embedder.py",
        "desc":    "Generate multilingual embeddings (GPU T4)",
        "output":  "embeddings.npy",
        "time":    "~5 min",
    },
    {
        "number":  5,
        "name":    "FAISS Builder",
        "file":    "gem_faiss_builder.py",
        "command": "modal run gem_faiss_builder.py",
        "desc":    "Build FAISS vector index",
        "output":  "faiss_index.bin + chunks_metadata.json",
        "time":    "~3 min",
    },
    {
        "number":  6,
        "name":    "Retriever Test",
        "file":    "gem_retriever.py",
        "command": "modal run gem_retriever.py",
        "desc":    "Test semantic retrieval (AR + EN queries)",
        "output":  "console output",
        "time":    "~2 min",
    },
]


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def run_command(cmd: str, step_name: str) -> bool:
    """Run a shell command, stream output, return success."""
    print(f"\n{'─'*60}")
    print(f"▶  Running: {cmd}")
    print(f"{'─'*60}")

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            text=True,
        )
        elapsed = round(time.time() - start, 1)
        print(f"\n✅ {step_name} completed in {elapsed}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = round(time.time() - start, 1)
        print(f"\n❌ {step_name} FAILED after {elapsed}s")
        print(f"   Exit code: {e.returncode}")
        return False


def print_banner():
    print("\n" + "=" * 60)
    print("🏛️  GEM RAG Pipeline — Master Runner")
    print("=" * 60)
    print("Steps:")
    for step in STEPS:
        print(
            f"  {step['number']}. {step['name']:<15} "
            f"{step['time']:<10} → {step['output']}"
        )
    print("=" * 60)


def check_files():
    """Check which step files exist."""
    print("\n📁 Script files:")
    all_ok = True
    for step in STEPS:
        exists = Path(step["file"]).exists()
        status = "✅" if exists else "❌ MISSING"
        print(f"   {status} {step['file']}")
        if not exists:
            all_ok = False
    return all_ok


# ─────────────────────────────────────────────
# Pipeline Runner
# ─────────────────────────────────────────────
def run_pipeline(from_step: int = 1, only_step: int = None):
    print_banner()

    if not check_files():
        print("\n❌ Some script files are missing. Aborting.")
        sys.exit(1)

    steps_to_run = (
        [s for s in STEPS if s["number"] == only_step]
        if only_step
        else [s for s in STEPS if s["number"] >= from_step]
    )

    if not steps_to_run:
        print(f"\n❌ No steps to run.")
        sys.exit(1)

    print(f"\n🚀 Running {len(steps_to_run)} step(s)...")

    failed = []
    for step in steps_to_run:
        print(f"\n{'='*60}")
        print(f"STEP {step['number']}/6: {step['name'].upper()}")
        print(f"  {step['desc']}")
        print(f"  Expected time: {step['time']}")
        print(f"{'='*60}")

        ok = run_command(step["command"], step["name"])
        if not ok:
            failed.append(step["name"])
            print(f"\n⚠️  Step {step['number']} failed.")
            choice = input("Continue anyway? [y/N]: ").strip().lower()
            if choice != "y":
                print("Aborting pipeline.")
                break

    # Summary
    print(f"\n{'='*60}")
    if not failed:
        print("🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
        print("\nNext steps:")
        print("  Deploy API  : python gem_run_all.py --deploy")
        print("  Launch UI   : python gem_run_all.py --ui")
        print("  Evaluate    : python gem_run_all.py --eval")
    else:
        print(f"⚠️  Completed with {len(failed)} failure(s):")
        for f in failed:
            print(f"   ❌ {f}")
    print(f"{'='*60}")


def deploy_api():
    """Deploy the pipeline as a Modal web endpoint."""
    print("\n🚀 Deploying GEM API to Modal...")
    print("   This creates a persistent web endpoint.")
    print("   After deploy, copy the URL into gem_streamlit.py\n")
    ok = run_command("modal deploy gem_pipeline.py", "API Deploy")
    if ok:
        print("\n✅ API deployed!")
        print("   The endpoint URL was printed above.")
        print("   Set it in gem_streamlit.py as MODAL_ASK_URL")


def launch_ui():
    """Launch Streamlit locally."""
    print("\n🖥️  Launching Streamlit UI...")
    print("   Make sure MODAL_ASK_URL is set in gem_streamlit.py first.")
    try:
        subprocess.run(
            "streamlit run gem_streamlit.py",
            shell=True,
            check=True,
        )
    except KeyboardInterrupt:
        print("\nStreamlit stopped.")


def run_eval():
    """Run the evaluator against the Modal endpoint."""
    print("\n🧪 Running Evaluation...")
    print("   Make sure MODAL_ASK_URL is set correctly.")
    try:
        subprocess.run(
            f"{sys.executable} gem_evaluator.py",
            shell=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        print("❌ Evaluation failed.")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="GEM RAG Pipeline Master Runner"
    )
    parser.add_argument(
        "--from",
        dest="from_step",
        type=int,
        default=1,
        help="Start from step N (1–6)",
    )
    parser.add_argument(
        "--step",
        dest="only_step",
        type=int,
        default=None,
        help="Run only step N",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Deploy API to Modal",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch Streamlit UI",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run evaluator",
    )

    args = parser.parse_args()

    if args.deploy:
        deploy_api()
    elif args.ui:
        launch_ui()
    elif args.eval:
        run_eval()
    else:
        run_pipeline(
            from_step=args.from_step,
            only_step=args.only_step,
        )


if __name__ == "__main__":
    main()