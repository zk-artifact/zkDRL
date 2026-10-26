# Anonymity checklist

This review copy was assembled from the final experimental source tree with the
following removals and normalizations.

## Removed

- Git repositories and history (`.git`, nested empty repositories).
- Author, institution, laboratory, grant and acknowledgment text.
- Cloud/rental orchestration, provider instance IDs, hostnames, IP addresses,
  SSH keys, `known_hosts`, API-token handling and billing records.
- Raw campaign directories: proof bundles, telemetry, auditor SQLite files,
  per-request stdout/stderr, collected cloud payloads, and runtime archives.
- Python caches, compiled host binaries and guest ELFs. Binaries can embed
  absolute build paths; they are rebuilt locally from source instead.
- Absolute development workspace paths from source and documentation.

## Retained deliberately

- Public upstream URLs for SP1, Rust crates, MiniWoB++ and KuaiRec.
- Public dataset provenance, hashes and the KuaiRec CC BY 4.0 notice.
- Sanitized aggregate tables in `reported-results/`; these contain configuration
  IDs, seeds, timings and resource peaks, but no account, host or path metadata.
  Figures are intentionally not shipped and can be regenerated locally.
- The code's own campaign name (`full-023-unified`) because it identifies the
  paper's single cohort, not an author or institution.

## Checks

```sh
bash scripts/check_anonymity.sh
```

The script scans text files for absolute workspace paths, development campaign
labels, email addresses, provider credentials, SSH material, GPU/instance IDs
and author/institution phrases. It scans hidden files and previously ignored
paths, and includes `reported-results/` text files. It excludes Cargo lockfiles
and binary fixtures from the text scan; those were reviewed separately. No PDFs
or compiled binaries are shipped; the small NPZ data fixture is scanned with
`strings`. The package script uses a fixed timestamp and sorted
entries, so the archive itself does not carry the original file mtimes.

## Known residual signals

- The Rust source and public documentation retain upstream project names and
  technical vocabulary (SP1, Groth16, MiniWoB++, KuaiRec); these are not
  author-identifying.
- The datasets' licenses and citations name their own authors; that is required
  attribution, not disclosure of the submission's authors.

If a residual identifier is found, report it through the anonymous review
system rather than editing the artifact in place.
