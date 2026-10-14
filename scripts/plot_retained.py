"""Isolated optional plotting route; reuses the exact existing figure functions.

Reads retained processed tables only. Does not run the exporter, experiment,
paper-copy steps, or any network command. Font licenses stay with the caller's
installed Matplotlib font distribution; no font binaries are redistributed.
"""
from pathlib import Path
from fractions import Fraction as Q
import argparse
import csv
import importlib.util
import math
import os
import warnings

ARTIFACT = Path(__file__).resolve().parents[1]


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))

def validate_inputs(source):
    fiber = read_rows(source / "fiber_curve.csv")
    if len(fiber) != 31:
        raise ValueError("Expected all thirty-one retained fiber samples")
    for i, row in enumerate(fiber):
        t = -Q(1, 20) + Q(i, 400)
        full = Q(9, 80)
        deferred = Q(17, 240) + Q(1, 24) + t/6
        for key, value in (("t", t), ("full", full), ("deferred", deferred),
                           ("bayes", min(full, deferred))):
            if Q(row[key]) != value or float(row[key + "_float"]) != float(value):
                raise ValueError(f"Fiber-coordinate mismatch at row {i}: {key}")
    fee = read_rows(source / "fee_deficiency.csv")
    if len(fee) != 21:
        raise ValueError("Expected all twenty-one retained fee samples")
    for i, row in enumerate(fee):
        price = Q(3, 80) + Q(i, 1600)
        alpha, beta = Q(1, 20) - price, price - Q(3, 80)
        expected = (("pair_fee", price), ("deterministic", min(alpha, beta)),
                    ("randomized", 80*alpha*beta))
        for key, value in expected:
            if Q(row[key]) != value or float(row[key + "_float"]) != float(value):
                raise ValueError(f"Fee-coordinate mismatch at row {i}: {key}")
        if Q(row["defer_probability"]) != 80*alpha:
            raise ValueError(f"Mixture probability mismatch at row {i}")
    return fiber, fee


def configure_fonts(renderer, font_dir):
    # Pin every CM face in-process, including mathtext; never alter font files.
    import matplotlib
    from matplotlib import font_manager, pyplot as plt
    renderer.configure_fonts()
    directory = font_dir or Path(matplotlib.get_data_path()) / "fonts/ttf"
    names = ("cmb10.ttf", "cmex10.ttf", "cmmi10.ttf", "cmr10.ttf",
             "cmss10.ttf", "cmsy10.ttf", "cmtt10.ttf")
    families = {}
    for name in names:
        path = (directory / name).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        family = font_manager.FontProperties(fname=str(path)).get_name()
        if family.casefold() != Path(name).stem.casefold():
            raise ValueError(f"Expected native Computer Modern face in {path}")
        families[family] = path
    font_manager.fontManager.ttflist = [
        item for item in font_manager.fontManager.ttflist if item.name not in families]
    for family, path in families.items():
        font_manager.fontManager.addfont(str(path))
        selected = font_manager.findfont(font_manager.FontProperties(family=family),
                                         fallback_to_default=False)
        if Path(selected).resolve() != path:
            raise ValueError(f"Font selection drift: {family}")
    plt.rcParams["mathtext.fallback"] = None


def isolated_output(destination, source):
    destination = destination.resolve()
    protected = (source.resolve(), (ARTIFACT / "results").resolve(),
                 (ARTIFACT / "figures").resolve(), (ARTIFACT.parent / "paper").resolve())
    if any(destination == root or root in destination.parents for root in protected):
        raise ValueError("Output must be separate from retained inputs, figures, and paper")
    destination.mkdir(parents=True, exist_ok=True)
    cache = destination / ".render-cache"
    cache.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    os.environ["TMP"] = str(cache)
    os.environ["TEMP"] = str(cache)
    return destination


def load_renderer():
    spec = importlib.util.spec_from_file_location(
        "retained_figure_renderer", ARTIFACT / "scripts/plot_figures.py")
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    return renderer


def save_private(fig, destination, name):
    from matplotlib import pyplot as plt
    target = destination / (name + ".pdf")
    fig.savefig(target, metadata={"Creator": None, "Producer": None,
                                  "CreationDate": None, "ModDate": None})
    plt.close(fig)
    print(f"Rendered vector figure: {target}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ARTIFACT / "results")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--font-dir", type=Path)
    args = parser.parse_args()
    validate_inputs(args.input_dir)
    destination = isolated_output(args.output_dir, args.input_dir)
    warnings.filterwarnings("error", message=r"Glyph .* missing")
    renderer = load_renderer()
    configure_fonts(renderer, args.font_dir)
    renderer.rows = lambda destination, name: read_rows(args.input_dir / name)
    renderer.save = save_private
    renderer.fiber_risks(destination)
    renderer.fee_deficiency(destination)


if __name__ == "__main__":
    main()
