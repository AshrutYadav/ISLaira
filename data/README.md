---
license: cc-by-nc-sa-4.0
extra_gated_fields:
  Full Name: text
  Affiliation (Organization/University): text
  Designation/Status in Your Organization: text
  Country: country
  I want to use this dataset for (please provide the reason(s)): text
  The iSign dataset is free for research use but NOT for commercial use; do you agree if you are provided with the iSign dataset, you will NOT use it for any commercial purposes? Also, do you agree that you will not be sharing this dataset further or uploading it anywhere else on the internet: checkbox
  DISCLAIMER The dataset is released for research purposes only, and authors do not take any responsibility for any damage or loss arising due to the usage of data or any system/model developed using the dataset: checkbox
tags:
- indian sign language
- machine translation
- sign language translation
size_categories:
- 100K<n<1M
pretty_name: iSign
configs:
- config_name: iSign_v1.1
  data_files: iSign_v1.1.csv
  default: true
- config_name: word-presence-dataset_v1.1
  data_files: word-presence-dataset_v1.1.csv
- config_name: word-description-dataset_v1.1
  data_files: word-description-dataset_v1.1.csv
task_categories:
- translation
---
# iSign: A Benchmark for Indian Sign Language Processing

The iSign dataset serves as a benchmark for Indian Sign Language Processing. The dataset comprises of NLP-specific tasks (including SignVideo2Text, SignPose2Text, Text2Pose, Word Prediction, and Sign Semantics). The dataset is free for research use but not for commercial purposes.

## Quick Links

- [**Website**](https://exploration-lab.github.io/iSign/): The landing page for iSign
- [**arXiv Paper**](https://arxiv.org/abs/2407.05404v1): Detailed information about the iSign Benchmark.
- [**Dataset on Hugging Face**](https://huggingface.co/datasets/Exploration-Lab/iSign/): Hugging Face link to get/download the iSign dataset.

## Dataset Usage

### Videos
The iSign videos and the corresponding pose files are available in part files (due to huggingface cap on file sizes). The video part files `iSign-videos_v1.1_part_aa` and `iSign-videos_v1.1_part_ab` can be combined to get the complete video dataset zip file using the following command:
```
cat iSign-videos_v1.1_part_aa iSign-videos_v1.1_part_ab > iSign-videos_v1.1.zip
```

### Pose
Similarly, the pose part files `iSign-poses_v1.1_part_aa`, `iSign-poses_v1.1_part_ab`, `iSign-poses_v1.1_part_ac`, and `iSign-poses_v1.1_part_ad` can be combined to get the complete pose dataset zip file using the following command:
```
cat iSign-poses_v1.1_part_aa iSign-poses_v1.1_part_ab iSign-poses_v1.1_part_ac iSign-poses_v1.1_part_ad > iSign-poses_v1.1.zip
```

The pose files are saved using the [**pose-format** [https://github.com/sign-language-processing/pose]](https://github.com/sign-language-processing/pose). 
```bash
pip install pose-format
```

#### Reading `.pose` Files: 

To load a `.pose` file, use the `Pose` class.

```python
from pose_format import Pose

data_buffer = open("file.pose", "rb").read()
pose = Pose.read(data_buffer)

numpy_data = pose.body.data
confidence_measure  = pose.body.confidence
```

### Text
The translations for the videos are available in the CSV files. `iSign_v1.1.csv` contains the translations for the videos, `word-presence-dataset_v1.1.csv` contains the word presence dataset for Task 4 (Word Presence Prediction) in the paper, and `word-description-dataset_v1.1.csv` contains the word description dataset for Task-5 (Semantic Similarity Prediction) in the paper.

Each entry in the datasets is identified by a unique identifier (UID) structured as follows:
- Format: `[video_id]-[sequence_number]` 
- Example: `1782bea75c7d-7`
  - `1782bea75c7d`: Unique video ID
  - `-7`: Sequence number within the video

Note the sequence number in the UID indicates the order of the text within each video, allowing for proper reconstruction of the full translation or description.
For train/dev/test split, we recommend splitting using the video_id, i.e. keeping all the videos with a video_id in the same split. 
This will ensures that all segments (rows) belonging to a single video remain together in the same split, preventing data leakage and contamination. 



## Citing Our Work

If you find the iSign dataset beneficial, please consider citing our work:
```
@inproceedings{iSign-2024,
  title = "{iSign}: A Benchmark for Indian Sign Language Processing",
  author = "Joshi, Abhinav  and
    Mohanty, Romit  and
    Kanakanti, Mounika  and
    Mangla, Andesha  and
    Choudhary, Sudeep  and
    Barbate, Monali  and
    Modi, Ashutosh",
  booktitle = "Findings of the Association for Computational Linguistics: ACL 2024",
  month = aug,    
  year = "2024",
  address = "Bangkok, Thailand",
  publisher = "Association for Computational Linguistics", 
}
```