"""
Standalone plot generator - regenerate figures for any completed experiment.

Usage:
  python plot.py results/CIFAR_CNN
  python plot.py results/CIFAR_CNN --type training_curves
  python plot.py results/CIFAR_CNN results/CIFAR_CNN_005   # multiple runs
"""

import argparse
import os
import sys

sys.dont_write_bytecode = True
sys.path.append(os.path.dirname(__file__))

from core.visualization import (
    plot_training_curves,
    plot_client_accuracy,
    plot_data_distribution,
    plot_class_accuracy_heatmap,
    generate_all_plots,
)

PLOT_FNS = {
    "training_curves":        plot_training_curves,
    "client_accuracy":        plot_client_accuracy,
    "data_distribution":      plot_data_distribution,
    "class_accuracy_heatmap": plot_class_accuracy_heatmap,
}


def main():
    parser = argparse.ArgumentParser(
        description="Re-generate FastFL figures from a results directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dirs", nargs="+", metavar="RESULTS_DIR",
        help="One or more experiment output directories",
    )
    parser.add_argument(
        "--type", choices=list(PLOT_FNS.keys()), default=None,
        help="Generate only this plot type (default: all)",
    )
    args = parser.parse_args()

    for d in args.dirs:
        if not os.path.isdir(d):
            print(f"  [skip] not a directory: {d}")
            continue
        metrics_path = os.path.join(d, "metrics.json")
        if not os.path.exists(metrics_path):
            print(f"  [skip] no metrics.json found in {d}")
            continue

        print(f"\n  {d}")
        if args.type:
            out = PLOT_FNS[args.type](d)
            if out:
                print(f"    -> {out}")
        else:
            generate_all_plots(d)


if __name__ == "__main__":
    main()
