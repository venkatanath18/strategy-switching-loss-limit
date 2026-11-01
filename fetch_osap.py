"""Download the Open Source Asset Pricing data used by real_data.py.

Needs the openassetpricing package (pip install openassetpricing). Saves the
signal documentation and the long-short returns of every predictor as CSV in
data/osap/, so real_data.py itself only needs numpy.
"""

import os

import openassetpricing as oap

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "osap")


def main():
    os.makedirs(OUT, exist_ok=True)
    openap = oap.OpenAP()

    openap.dl_signal_doc("pandas").to_csv(
        os.path.join(OUT, "signal_doc.csv"), index=False)

    ports = openap.dl_port("op", "pandas")
    ports = ports[ports["port"] == "LS"][["signalname", "date", "ret"]]
    ports.to_csv(os.path.join(OUT, "ls_returns.csv"), index=False)

    print(f"Saved {ports['signalname'].nunique()} strategies to {OUT}")


if __name__ == "__main__":
    main()
