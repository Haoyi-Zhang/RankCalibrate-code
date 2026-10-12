#!/usr/bin/env python3
"""Draw the retained exact fiber and deficiency tables as vector PDFs.

Requires Matplotlib with its bundled Computer Modern TTF fonts. No TeX,
experiment runner, private helper, or host-specific font path is required.
"""
from pathlib import Path
import csv
import shutil

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt

ARTIFACT = Path(__file__).resolve().parents[1]



def correct_cm_unicode():
    """Keep legacy CM glyphs but write their real Unicode PDF text mappings.

    Matplotlib's CM math fonts use TeX character slots, not Unicode slots.
    Correct only ToUnicode streams; vector paths, glyph shapes, and metrics
    remain untouched. FontTools is already a Matplotlib dependency.
    """
    from fontTools import agl
    from fontTools.ttLib import TTFont
    from matplotlib.backends.backend_pdf import PdfFile

    embed = PdfFile.embedTTF
    output = PdfFile.outputStream

    def write_stream(pdf, reference, data, *args, **kwargs):
        mapping = getattr(pdf, "_cm_text_mapping", None)
        if mapping is not None and isinstance(data, bytes) and b"begincmap" in data:
            entries = [
                f"<{code:04x}> <{value.encode('utf-16-be').hex()}>"
                for code, value in sorted(mapping.items())
            ]
            groups = [
                f"{len(entries[i:i+100])} beginbfchar\n"
                + "\n".join(entries[i:i+100]) + "\nendbfchar"
                for i in range(0, len(entries), 100)
            ]
            data = (
                "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
                "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
                "/CMapName /ComputerModernUnicode def\n/CMapType 2 def\n"
                "1 begincodespacerange\n<0000> <ffff>\nendcodespacerange\n"
                + "\n".join(groups)
                + "\nendcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n"
            ).encode("ascii")
        return output(pdf, reference, data, *args, **kwargs)

    def embed_font(pdf, filename, characters):
        mapping = None
        if Path(filename).name in ("cmr10.ttf", "cmmi10.ttf", "cmsy10.ttf", "cmex10.ttf"):
            with TTFont(filename) as font:
                cmap = font.getBestCmap()
                mapping = {code: agl.toUnicode(cmap[code]) for code in characters}
                if any(not value for value in mapping.values()):
                    raise ValueError("No Unicode name for a used Computer Modern glyph")
        pdf._cm_text_mapping = mapping
        try:
            return embed(pdf, filename, characters)
        finally:
            pdf._cm_text_mapping = None

    PdfFile.embedTTF = embed_font
    PdfFile.outputStream = write_stream

def configure_fonts():
    correct_cm_unicode()
    fonts = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for name in ("cmr10.ttf", "cmmi10.ttf", "cmsy10.ttf", "cmex10.ttf"):
        path = fonts / name
        if not path.is_file():
            raise FileNotFoundError(f"Matplotlib's bundled Computer Modern font: {name}")
        font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({"font.family": "cmr10", "font.size": 10.91,
                         "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
                         "axes.unicode_minus": False, "text.usetex": False,
                         "pdf.fonttype": 42, "pdf.use14corefonts": False,
                         "axes.linewidth": 0.8, "xtick.labelsize": 9.96,
                         "ytick.labelsize": 9.96})


def rows(destination, name):
    source = ARTIFACT / "results" / name
    target = destination / name
    shutil.copyfile(source, target)
    consumer = ARTIFACT.parent / "paper" / "figures"
    if consumer.is_dir():
        shutil.copyfile(target, consumer / name)
    with target.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def save(fig, destination, name):
    target = destination / f"{name}.pdf"
    fig.savefig(target, metadata={"Creator": None, "Producer": None,
                                  "CreationDate": None, "ModDate": None})
    plt.close(fig)
    consumer = ARTIFACT.parent / "paper" / "figures"
    if consumer.is_dir():
        shutil.copyfile(target, consumer / target.name)
    print(f"Rendered vector figure: {target}")


def fiber_risks(destination):
    data = rows(destination, "fiber_curve.csv")
    fig = plt.figure(figsize=(378 / 72, 144 / 72))
    ax = fig.add_axes((49 / 378, 29 / 144, 319 / 378, 108 / 144))
    for key, style, label in (("full_float", "-", r"Complete ranking $F$"),
                              ("deferred_float", "--", r"Resolved pair $D$")):
        ax.plot([float(row["t_float"]) for row in data],
                [float(row[key]) for row in data], color="black",
                linewidth=1.0, linestyle=style, label=label)
    ax.set(xlim=(-0.051, 0.026), ylim=(0.102, 0.119),
           xlabel=r"Fiber parameter $t$", ylabel="Expected priced loss")
    ax.set_xticks((-0.05, -0.025, 0, 0.025),
                 (r"$-0.05$", r"$-0.025$", "0", "0.025"))
    ax.set_yticks((0.105, 0.110, 0.115), ("0.105", "0.110", "0.115"))
    finish_axes(ax)
    save(fig, destination, "fiber-risks")


def fee_deficiency(destination):
    data = rows(destination, "fee_deficiency.csv")
    fig = plt.figure(figsize=(386 / 72, 152 / 72))
    ax = fig.add_axes((64 / 386, 33 / 152, 319 / 386, 108 / 152))
    for key, style, label in (("deterministic_float", "-", "Deterministic"),
                              ("randomized_float", "--", "Randomized")):
        ax.plot([float(row["pair_fee_float"]) for row in data],
                [float(row[key]) for row in data], color="black",
                linewidth=1.0, linestyle=style, label=label)
    ax.set(xlim=(0.037, 0.0505), ylim=(0, 0.0075),
           xlabel=r"Uniform pair fee $\lambda$", ylabel="Worst-fiber regret")
    ax.set_xticks((0.0375, 0.04166667, 0.04375, 0.05),
                 (r"$3/80$", r"$1/24$", r"$7/160$", r"$1/20$"))
    ax.set_yticks((0, 0.0025, 0.005, 0.0075),
                 ("0", r"$2.5\cdot 10^{-3}$", r"$5\cdot 10^{-3}$", r"$7.5\cdot 10^{-3}$"))
    finish_axes(ax)
    save(fig, destination, "fee-deficiency")


def finish_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=2.5, pad=3)
    ax.legend(loc="upper left", frameon=False, fontsize=9.96,
              borderaxespad=0.3, handlelength=1.7,
              handletextpad=0.4, labelspacing=0.2)


def main():
    configure_fonts()
    destination = ARTIFACT / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    fiber_risks(destination)
    fee_deficiency(destination)


if __name__ == "__main__":
    main()
