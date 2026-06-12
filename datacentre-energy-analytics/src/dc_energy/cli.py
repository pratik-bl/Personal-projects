"""Command-line interface for the dc-energy pipeline.

Examples
--------
dc-energy clean-workstation data/samples/workstation_may_aug_2021_raw_SAMPLE.csv -o ws.parquet
dc-energy clean-worldbank data/samples/World_bank_population.csv -o pop.parquet
dc-energy clean-dc-counts data/samples/data-centres-worldwide-general-dataset.csv -o dc.parquet
dc-energy density data/processed/global_dc_counts_tidy.parquet data/processed/world_pop_long.parquet -o density.csv
dc-energy anchors data/samples/workstation_2021_may_aug_tidy_SAMPLE.parquet
dc-energy train data/samples/*_tidy_SAMPLE.parquet -o models/surrogate.txt
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def _save(df: pd.DataFrame, out: str | None):
    if out is None:
        print(df.head(10).to_string())
        print(f"... {len(df):,} rows (use -o to save)")
        return
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".csv":
        df.to_csv(out, index=False)
    else:
        df.to_parquet(out, index=False)
    print(f"wrote {out}  ({len(df):,} rows)")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="dc-energy",
                                     description="Data-centre energy analytics pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("clean-workstation", help="Clean raw 2021 telemetry CSV")
    p.add_argument("path")
    p.add_argument("-o", "--out")

    p = sub.add_parser("clean-worldbank", help="Clean World Bank population CSV")
    p.add_argument("path")
    p.add_argument("-o", "--out")

    p = sub.add_parser("clean-dc-counts", help="Clean scraped DC counts CSV")
    p.add_argument("path")
    p.add_argument("-o", "--out")

    p = sub.add_parser("clean-ukpn", help="Clean UKPN demand-profile export")
    p.add_argument("path")
    p.add_argument("-o", "--out")

    p = sub.add_parser("clean-seds", help="Clean a US EIA SEDS file")
    p.add_argument("path")
    p.add_argument("--tag", required=True, help="dataset label, e.g. co2, prices")
    p.add_argument("-o", "--out")

    p = sub.add_parser("density", help="Data-centres per million population")
    p.add_argument("dc_counts")
    p.add_argument("population")
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--drop-top-n", type=int, default=0)
    p.add_argument("-o", "--out")

    p = sub.add_parser("anchors", help="Server power anchors from telemetry parquets")
    p.add_argument("paths", nargs="+")

    p = sub.add_parser("train", help="Train LightGBM power surrogate")
    p.add_argument("paths", nargs="+")
    p.add_argument("-o", "--out", help="model output path (.txt)")
    p.add_argument("--rounds", type=int, default=400)
    p.add_argument("--folds", type=int, default=5)

    args = parser.parse_args(argv)

    if args.command == "clean-workstation":
        from .cleaning import tidy_workstation
        _save(tidy_workstation(args.path), args.out)

    elif args.command == "clean-worldbank":
        from .cleaning import tidy_world_bank_pop
        _save(tidy_world_bank_pop(args.path), args.out)

    elif args.command == "clean-dc-counts":
        from .cleaning import tidy_global_dc
        _save(tidy_global_dc(args.path), args.out)

    elif args.command == "clean-ukpn":
        from .cleaning import tidy_ukpn_timeseries
        _save(tidy_ukpn_timeseries(args.path), args.out)

    elif args.command == "clean-seds":
        from .cleaning import tidy_seds
        _save(tidy_seds(args.path, args.tag), args.out)

    elif args.command == "density":
        from .analysis import dc_per_capita
        dc = pd.read_parquet(args.dc_counts)
        pop = pd.read_parquet(args.population)
        _save(dc_per_capita(dc, pop, year=args.year, drop_top_n=args.drop_top_n), args.out)

    elif args.command == "anchors":
        from .analysis import server_power_anchors
        print(json.dumps(server_power_anchors(args.paths), indent=2))

    elif args.command == "train":
        from .modeling import train_surrogate
        model, rmses = train_surrogate(
            args.paths, out_path=args.out,
            num_boost_round=args.rounds, n_folds=args.folds,
        )
        print(json.dumps({"cv_rmse": rmses, "mean_cv_rmse": sum(rmses) / len(rmses)}, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
