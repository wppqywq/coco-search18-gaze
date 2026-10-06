# Third-party provenance

The local `gbvs-master/saliency_models` directory is a subset of
[shreelock/gbvs](https://github.com/shreelock/gbvs), revision
`4c7e3c58e786d053ec4ae6fb9d03bdfeea2c539a`. It is a Python port, not the
reference MATLAB implementation. The local snapshot has no license file.
Its code and resource files are excluded from the publication snapshot.

Local corrections handle constant/zero maps and fix the Gabor coordinate
sign. The frozen feature manifest hashes all Python files, including these
corrections. An unmodified upstream checkout is not a drop-in replacement.
The provenance checks intentionally reject mismatched sources.

This dependency remains at its original local path to preserve existing
verification. Historical third-party sources are retained only in the local
recovery archive; they are not included in the public export. The public
release supports inspection and figure reproduction without GBVS, but full
feature reproduction requires the exact original assets and appropriate
redistribution permissions.
