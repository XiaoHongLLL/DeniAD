# Public log datasets

Raw public logs are intentionally not redistributed.

- HDFS, BGL, Thunderbird: https://github.com/logpai/loghub
- Spirit, Liberty: https://www.usenix.org/cfdr-data

After downloading the datasets under their original terms, use:

```bash
python dataset/prepare/prepare_labeled_log_datasets.py \
  --raw_root /path/to/raw_logs \
  --out_root ./data \
  --datasets hdfs bgl thunderbird spirit liberty \
  --seed 2023
```

Record the download date, source release, checksums, and any sampling limits in
the final artifact metadata.

