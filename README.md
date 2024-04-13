This section outlines how to prepare a valid submission using the runtime environment.

Prerequisites
Ensure you have:

Cloned this repository.
At least 10 GB free for the dataset and 5 GB for Docker images.
Docker installed (Installation Guide).
GNU Make installed (Installation Guide), optional but recommended.
Download the Data
Access the competition's data download page. Download and place submission_format.csv and training_labels.csv in the /data directory of this project. Download and extract at least one tar file of image data to data/images.

Your directory structure should be:

kotlin
Copy code
data/
├── images/
│   ├── example1.png
│   ├── example2.png
│   └── ...
├── submission_format.csv
└── train_labels.csv
Running the Quickstart Example
Navigate to the example_src/ directory in this repository to find a simple example. This example won't win the competition but will help you get started with your submission:

It includes a main.sh script, which sets path variables and runs main.py.
The main.py script generates arbitrary bounding boxes for each image, ensuring you understand the submission format.
Test Your Submission
Use the Docker setup to simulate the competition's runtime environment:

Pull the latest Docker image:
bash
Copy code
make pull
Prepare your submission by zipping your solution:
bash
Copy code
make clean && make pack-example
Test your submission:
bash
Copy code
make test-submission
This will generate a submission.csv under submission/.
Developing Your Submission
Beyond the quickstart, here’s how to develop your tailored solution:

Set up your environment following the Prerequisites.
Download data as detailed in Download the Data.
Develop your solution in the submission_src directory.
Use the provided Makefile for routine tasks:
bash
Copy code
make pull
make pack-submission
make test-submission
Test locally to ensure your submission performs as expected.
Local Evaluation
Use the scripts/score.py to calculate the Jaccard index score locally:

bash
Copy code
python scripts/score.py data/submission_format.csv data/train_labels.csv
This helps adjust your models based on the competition's scoring metric.

