VocalBridge: Indian Sign Language (ISL) Translation Benchmark
VocalBridge is an end-to-end machine translation system developed to bridge the communication gap for the hard-of-hearing community by converting Indian Sign Language (ISL) into English text. The project utilizes the iSign and ISLTranslate datasets, which provide a comprehensive collection of video and pose-based resources for continuous sign language processing.

Project Overview
This repository focuses on Sign-to-Text translation using deep learning architectures. By processing skeleton-based pose data, the system translates sequence-based gestures into natural language transcripts.

Dataset Specifications
The project handles a massive data scale of approximately 170GB, distributed across four main parts.

Core Dataset: iSign v1.1, comprising over 100,000 sequences.

Data Format: Pose files are stored in a specialized .pose format, which includes body data and confidence measures.

Task Focus: Primarily SignPose2Text translation and Sign Semantics.

Technical Setup & Workflow
1. Local Server Configuration
Due to the specific hardware constraints of our local infrastructure (Intel Core 2 Quad Q9600), the environment is strictly optimized to avoid instruction set errors.

Compiler Target: core2 (Penryn architecture).

Python Version: 3.9 (cp39).

Dependency Locking: Protobuf is capped at 3.20.x and NumPy is restricted to < 2.0 to maintain compatibility between MediaPipe and older TensorFlow builds.
download your suitable tensorflow version 2.8.0 at :https://github.com/yaroslavvb/tensorflow-community-wheels/issues/209

2. Environment Initialization
Run the following commands to set up the local development environment:

Bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name vocalbridge_env --display-name "VocalBridge (Python 3.9)"
3. Kaggle Training Workflow
Main model training is conducted on Kaggle to utilize modern GPU accelerators.

Data Access: Data is split into four parts (part_aa through part_ad) to bypass disk limits.

Loading Strategy: A custom Python Generator is used to stream data directly from the input directories, preventing memory overflow.

Repository Structure
notebooks/: Contains exploration and prototyping files.

src/: Production-ready Python scripts for data loading and training.

assets/: Project images and documentation resources.

requirements.txt: Pinned dependencies for team-wide synchronization.

Team Members
Ashmit Garg

Rehant Ukale

Dhruv Gupta

Yash

Atharva Mule

Citation and Licensing
This project utilizes resources from the Exploration-Lab. If you use this code or the associated datasets, please cite the following:

Code snippet
@inproceedings{iSign-2024,
  title = "{iSign}: A Benchmark for Indian Sign Language Processing",
  author = "Joshi, Abhinav and Mohanty, Romit and Kanakanti, Mounika and Mangla, Andesha and Choudhary, Sudeep and Barbate, Monali and Modi, Ashutosh",
  booktitle = "Findings of the Association for Computational Linguistics: ACL 2024",
  year = "2024"
}

@inproceedings{joshi-etal-2023-isltranslate,
    title = "{ISLT}ranslate: Dataset for Translating {I}ndian {S}ign {L}anguage",
    author = "Joshi, Abhinav and Agrawal, Susmit and Modi, Ashutosh",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    year = "2023"
}
The iSign and ISLTranslate datasets are released under the CC BY-NC-SA 4.0 license and are intended for research purposes only.
