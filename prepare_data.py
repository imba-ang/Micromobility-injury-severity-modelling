from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import sys

BASE = Path(__file__).resolve().parent
ROOT = Path(sys.argv[1])
TABLE2 = [
    "sp20",
    "sp30",
    "sp60",
    "morning",
    "afternoon",
    "evening",
    "night",
    "daylght",
    "dry_rd",
    "wet_rd",
    "rain_wht",
    "fine_wht",
    "urbn_rd",
    "rdabout",
    "gog_ahd",
    "mov_off",
    "ovt_mov",
    "jun_app",
    "jun_main",
    "jun_md_jn",
    "jun_exit",
    "no_skid",
    "pd_cwy",
    "pd_ftpth",
    "ln_main",
    "no_pd_50",
    "slw_stop",
    "ovt_stat",
    "imp_frnt",
    "jry_wrk",
    "male",
    "female",
    "ag_1620",
    "ag_2125",
    "ag_2635",
    "hm_town",
    "ped_feml",
    "pd_1115",
    "pd_2635",
    "pd_3745",
    "pd_4655",
    "pd_ov55",
]


def main():
    (BASE / "prepared").mkdir(exist_ok=True)
    (BASE / "results").mkdir(exist_ok=True)
    flows = []
    sources = []
    stats = []
    for mode in ["cyclist", "escooter"]:
        for year in [2022, 2023, 2024]:
            path = ROOT / f"{mode}{year}.xlsx"
            d = pd.read_excel(path).copy()
            d = d.rename(columns={c: c.replace("accident_", "collision_") for c in d})
            sources.append(
                {
                    "file": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "raw_n": len(d),
                }
            )
            d["excel_row"] = np.arange(len(d)) + 2
            keep = d.casualty_severity.isin([1, 2, 3])
            for c in [
                "collision_year_vehicle",
                "collision_year_collision",
                "collision_year",
            ]:
                keep &= d[c].eq(year)

            def canon(a):
                return a.astype("string").str.replace(r"\.0$", "", regex=True)

            for c in ["collision_index_collision", "collision_index"]:
                keep &= canon(d[c]).eq(canon(d.collision_index_vehicle)).fillna(False)
            d = d.loc[keep].copy()
            assert d.casualty_severity.isin([1, 2, 3]).all()
            if "gog_ahd" not in d:
                d["gog_ahd"] = d.vehicle_manoeuvre_historic.eq(18).astype(int)
                native = d.vehicle_manoeuvre_historic.eq(-1)
                d.loc[native, "gog_ahd"] = (
                    d.loc[native, "vehicle_manoeuvre"].eq(19).astype(int)
                )
            if "pd_ftpth" not in d:
                d["pd_ftpth"] = d.pedestrian_location.eq(6).astype(int)
            if "pd_cwy" not in d:
                d["pd_cwy"] = d.pedestrian_location.eq(5).astype(int)
            for col in ["hm_urbn", "hm_town"]:
                if col not in d:
                    d[col] = np.nan
            d["junction_detail_0"] = d.junction_detail.eq(0).astype(int)
            d["first_point_of_impact_3"] = d.first_point_of_impact.eq(3).astype(int)
            cols = list(
                dict.fromkeys(
                    TABLE2 + ["weekdy", "junction_detail_0", "first_point_of_impact_3"]
                )
            )
            frame = d[cols].copy()
            frame["severity"] = d.casualty_severity.isin([1, 2]).astype(int)
            frame["year"] = year
            frame["mode"] = mode
            frame["excel_row"] = d.excel_row
            frame.to_csv(BASE / "prepared" / f"{mode}{year}.csv", index=False)
            flows.append(
                dict(
                    mode=mode,
                    year=year,
                    raw_n=sources[-1]["raw_n"],
                    analysis_n=len(frame),
                    severe=int(frame.severity.sum()),
                    slight=int((1 - frame.severity).sum()),
                )
            )
            for col in [
                "severity",
                "sp30",
                "sp20",
                "sp60",
                "morning",
                "afternoon",
                "evening",
                "weekdy",
                "daylght",
                "rain_wht",
                "fine_wht",
                "dry_rd",
                "wet_rd",
                "urbn_rd",
                "rdabout",
                "mov_off",
                "ovt_mov",
                "jun_app",
                "jun_main",
                "jun_md_jn",
                "jun_exit",
                "no_skid",
                "no_pd_50",
                "pd_ftpth",
                "gog_ahd",
                "ln_main",
                "imp_frnt",
                "first_point_of_impact_3",
                "junction_detail_0",
                "male",
                "ag_1620",
                "ag_2125",
                "ag_2635",
                "jry_wrk",
                "pd_1115",
                "pd_2635",
                "pd_4655",
                "pd_3745",
                "pd_ov55",
                "ped_feml",
            ]:
                stats.append(
                    dict(
                        mode=mode,
                        year=year,
                        variable=col,
                        n=len(frame),
                        mean=frame[col].mean(),
                        sd=frame[col].std(ddof=1),
                    )
                )
            print(mode, year, len(frame), flush=True)
    pd.DataFrame(flows).to_csv(BASE / "results" / "sample_flow.csv", index=False)
    pd.DataFrame(stats).to_csv(BASE / "results" / "table2.csv", index=False)
    (BASE / "results" / "source_manifest.json").write_text(
        json.dumps(sources, indent=2)
    )


if __name__ == "__main__":
    main()
