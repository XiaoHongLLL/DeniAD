# Data and code availability text

## Submission-ready draft

The processed Train-Ticket Expected/Unexpected software-change dataset
generated in this study, including the frozen train, development and test
splits, run-level annotations, event-sequence inputs and metadata, is available
in [REPOSITORY] under [DOI OR PERSISTENT IDENTIFIER]. The corresponding DeniAD
implementation, data-construction scripts, experiment configurations and
evaluation code are available at [GITHUB URL] and archived as release
[VERSION] at [ARCHIVE DOI]. Public HDFS, BGL and Thunderbird logs were obtained
from Loghub; Spirit and Liberty logs were obtained from the USENIX Computer
Failure Data Repository. Raw third-party logs are not redistributed by the
authors and remain available from their original providers under the providers'
terms.

## Code availability text

The source code required to train and evaluate DeniAD and reproduce RQ1--RQ4 is
available at [GITHUB URL], release [VERSION], and archived at [ARCHIVE DOI]. The
archive contains the model implementation, preprocessing and dataset-building
code, frozen experiment scripts, environment specification and checksums. Raw
third-party logs and third-party baseline source trees are not redistributed.

## 作者上传后必须替换

- `[REPOSITORY]`：建议填写 Zenodo、Figshare、OSF 或学校长期数据仓库。
- `[DOI OR PERSISTENT IDENTIFIER]`：不能只填临时网盘链接。
- `[GITHUB URL]`、`[VERSION]`、`[ARCHIVE DOI]`：必须与 README 和论文一致。
- 如果 Train-Ticket 处理后数据只放 GitHub Release，建议再归档到 Zenodo，
  因为 GitHub 普通分支 URL 不是长期数据标识符。

