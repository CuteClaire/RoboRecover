# Third-party notices and release status

The existing repository Apache-2.0 license is retained. It does not override the terms of third-party code or grant a license to simulator assets, datasets, or model weights.

`overlays/` contains evaluation extensions and modified support files collected from the corresponding upstream checkouts. Preserve each component's original notices; available license files are in `third_party_licenses/`. `code_manifest.json` records source and release hashes. Path defaults were sanitized, SmolVLA's RoboTwin root made environment-configurable, and the optional OpenPI RLDS writer import made lazy.

Dependency revisions describe collected source bases, not proof that every historical evaluation used that revision. UniFOLM's revision was recovered from the full backup checkout; its evaluator matches the live snapshot by SHA256. License provenance still requires review. This snapshot is not a completed licensing or runtime certification.

Full scenario data redistribution terms and hosting are pending. The two example JSON files are provided for format inspection in this development snapshot; no separate dataset license has yet been selected by the authors.
