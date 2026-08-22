"""Extract and combine the 12 Beijing air-quality station CSV files."""

from __future__ import annotations

import csv
import io
import sys
import zipfile
from pathlib import Path


ACTIVITY_DIR = Path(__file__).parent
DATA_DIR = ACTIVITY_DIR / "data"
COMBINED_FILE = ACTIVITY_DIR / "beijing_air_quality_combined.csv"
DEFAULT_SOURCE = Path.home() / "Downloads" / "beijing+multi+site+air+quality+data.zip"
LOCAL_EXTRACTED_SOURCE = ACTIVITY_DIR / "beijing+multi+site+air+quality+data"
INNER_ZIP_NAME = "PRSA2017_Data_20130301-20170228.zip"


def extract_station_files(source_zip: Path) -> list[Path]:
    """Extract the station CSVs from the ZIP nested inside the UCI download."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(source_zip) as outer_zip:
        inner_bytes = outer_zip.read(INNER_ZIP_NAME)

    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner_zip:
        csv_names = sorted(
            name
            for name in inner_zip.namelist()
            if name.lower().endswith(".csv") and not name.startswith("__MACOSX")
        )
        if len(csv_names) != 12:
            raise ValueError(f"Expected 12 station CSV files, found {len(csv_names)}")

        extracted = []
        for member_name in csv_names:
            destination = DATA_DIR / Path(member_name).name
            destination.write_bytes(inner_zip.read(member_name))
            extracted.append(destination)

    return extracted


def combine_station_files(csv_files: list[Path]) -> tuple[int, int, set[str]]:
    """Combine station files while writing the header only once."""
    expected_header: list[str] | None = None
    row_count = 0
    stations: set[str] = set()

    with COMBINED_FILE.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)

        for csv_file in csv_files:
            with csv_file.open(encoding="utf-8-sig", newline="") as source:
                reader = csv.reader(source)
                header = next(reader)

                if expected_header is None:
                    expected_header = header
                    writer.writerow(header)
                elif header != expected_header:
                    raise ValueError(f"Column mismatch in {csv_file.name}")

                station_index = header.index("station")
                for row in reader:
                    if not row:
                        continue
                    writer.writerow(row)
                    row_count += 1
                    stations.add(row[station_index])

    return row_count, len(expected_header or []), stations


def main() -> None:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        LOCAL_EXTRACTED_SOURCE if LOCAL_EXTRACTED_SOURCE.exists() else DEFAULT_SOURCE
    )
    if not source.exists():
        raise FileNotFoundError(f"Dataset source not found: {source}")

    if source.is_dir():
        csv_files = sorted(source.rglob("PRSA_Data_*.csv"))
        if len(csv_files) != 12:
            raise ValueError(f"Expected 12 station CSV files, found {len(csv_files)}")
    else:
        csv_files = extract_station_files(source)
    row_count, column_count, stations = combine_station_files(csv_files)

    if row_count != 420_768:
        raise ValueError(f"Expected 420,768 rows, found {row_count:,}")
    if column_count != 18:
        raise ValueError(f"Expected 18 columns, found {column_count}")
    if len(stations) != 12:
        raise ValueError(f"Expected 12 stations, found {len(stations)}")

    print(f"Station files extracted: {len(csv_files)}")
    print(f"Combined rows: {row_count:,}")
    print(f"Columns: {column_count}")
    print(f"Stations: {', '.join(sorted(stations))}")
    print(f"Combined dataset: {COMBINED_FILE}")


if __name__ == "__main__":
    main()
