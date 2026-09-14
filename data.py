import csv
from pathlib import Path

def load_locations(file_path):
    names = []
    coordinates = []

    with Path(file_path).open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"{file_path} has no CSV header")

        if "city" in reader.fieldnames:
            name_field = "city"
            country_field = "country"
        elif "capital" in reader.fieldnames:
            name_field = "capital"
            country_field = "country"
        else:
            raise ValueError(f"{file_path} must contain either city or capital data")

        for row in reader:
            names.append(f"{row[name_field]}, {row[country_field]}")
            coordinates.append((float(row["latitude"]), float(row["longitude"])))

    return names, coordinates
