#!/usr/bin/env python3
"""Split merged KML tours into one KML file per gx:FlyTo frame."""

from __future__ import annotations

import argparse
import copy
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


KML_NS = "http://www.opengis.net/kml/2.2"
GX_NS = "http://www.google.com/kml/ext/2.2"
ATOM_NS = "http://www.w3.org/2005/Atom"
NS = {"kml": KML_NS, "gx": GX_NS}

ET.register_namespace("", KML_NS)
ET.register_namespace("gx", GX_NS)
ET.register_namespace("kml", KML_NS)
ET.register_namespace("atom", ATOM_NS)


def safe_id(value: str) -> str:
    """Return a single safe path component for a tour ID."""
    value = value.strip()
    if not value or value in {".", ".."} or re.search(r"[/\\\x00]", value):
        raise ValueError(f"invalid tour_id: {value!r}")
    return value


def split_file(source: Path, output_root: Path, overwrite: bool) -> int:
    tree = ET.parse(source)
    root = tree.getroot()
    tours = root.findall(".//gx:Tour", NS)
    if not tours:
        raise ValueError("no gx:Tour element found")

    frame_count = 0
    for tour in tours:
        name = tour.find("kml:name", NS)
        if name is None or not (name.text or "").strip():
            raise ValueError("gx:Tour is missing its tour_id in <name>")
        tour_id = safe_id(name.text or "")
        playlist = tour.find("gx:Playlist", NS)
        if playlist is None:
            raise ValueError(f"tour {tour_id!r} has no gx:Playlist")
        frames = playlist.findall("gx:FlyTo", NS)
        if not frames:
            raise ValueError(f"tour {tour_id!r} has no gx:FlyTo frames")

        tour_dir = output_root / tour_id
        tour_dir.mkdir(parents=True, exist_ok=True)

        for frame_id, frame in enumerate(frames):
            destination = tour_dir / f"{frame_id:04d}.kml"
            if destination.exists() and not overwrite:
                raise FileExistsError(
                    f"{destination} already exists; pass --overwrite to replace it"
                )

            output_root_element = ET.Element(root.tag, root.attrib)
            for child in root:
                if child is tour:
                    output_tour = copy.deepcopy(tour)
                    output_playlist = output_tour.find("gx:Playlist", NS)
                    for action in list(output_playlist):
                        output_playlist.remove(action)
                    output_playlist.append(copy.deepcopy(frame))
                    output_root_element.append(output_tour)
                else:
                    output_root_element.append(copy.deepcopy(child))

            ET.ElementTree(output_root_element).write(
                destination,
                encoding="utf-8",
                xml_declaration=True,
            )
            frame_count += 1

    return frame_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split every top-level KML gx:Tour into one file per gx:FlyTo."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/root/pairUAV_img/tours_merged_kml"),
        help="directory containing merged KML files (default: /root/pairUAV_img/tours_merged_kml)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/pairUAV_img/tours_splited_kml"),
        help="directory to write split tours (default: /root/pairUAV_img/tours_splited_kml)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing frame files",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        parser.error(f"input directory does not exist: {args.input_dir}")

    sources = sorted(
        path
        for path in args.input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".kml"
    )
    if not sources:
        parser.error(f"no .kml files found in: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_frames = 0
    failed_files = 0
    for source in sources:
        try:
            count = split_file(source, args.output_dir, args.overwrite)
            total_frames += count
            print(f"{source.name}: wrote {count} frames")
        except (ET.ParseError, OSError, ValueError) as error:
            failed_files += 1
            print(f"ERROR {source.name}: {error}", file=sys.stderr)

    print(
        f"Finished: {len(sources) - failed_files}/{len(sources)} files, "
        f"{total_frames} frames written to {args.output_dir}"
    )
    return 1 if failed_files else 0


if __name__ == "__main__":
    raise SystemExit(main())
