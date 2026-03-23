Summary of changes for tar archive support:

Files modified:
1. nanoresearch/agents/setup_search.py
   - Added imports: gzip, tarfile, zipfile
   - Added _extract_archive() method to handle .tar.gz, .tgz, .tar.bz2, .tar.xz, .txz, .gz, .zip
   - Added _get_decompressed_paths() helper method
   - Modified _download_resources() to extract archives after download

2. nanoresearch/agents/execution/repair_resources.py  
   - Added import: tarfile
   - Modified _materialize_missing_resource_targets() to extract tar archives when files are missing

Supported compression formats:
- .tar.gz / .tgz  -> extracted to directory
- .tar.bz2        -> extracted to directory  
- .tar.xz / .txz  -> extracted to directory
- .gz (non-tar)   -> decompressed to single file
- .zip            -> extracted to directory

Usage:
When nanoresearch downloads a dataset, it will now automatically:
1. Download the archive file
2. Detect the archive format
3. Extract it to the same data directory
4. Return the path to the extracted content

The extracted content is then used by the experiment code.
