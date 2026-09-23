# Third-party notices and release status

The existing repository Apache-2.0 license is retained. It does not override the terms of third-party code or grant a license to simulator assets, datasets, or model weights.

`overlays/` contains evaluation extensions and modified support files collected from the corresponding upstream checkouts. Preserve each component's original notices; available license files are in `third_party_licenses/`. `code_manifest.json` records source and release hashes. Path defaults were sanitized, SmolVLA's RoboTwin root made environment-configurable, and the optional OpenPI RLDS writer import made lazy.

Dependency revisions describe the collected local snapshots; they are not a claim that every historical evaluation used that revision. UniFOLM's base revision and license provenance remain unresolved, so the overlay installer intentionally refuses to install it. Do not interpret this development snapshot as a completed licensing or runtime certification.

Full scenario data redistribution terms and hosting are pending. The two example JSON files are provided for format inspection in this development snapshot; no separate dataset license has yet been selected by the authors.
