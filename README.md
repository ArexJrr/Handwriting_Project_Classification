# Handwriting ✍ Classification
<h1 align="center">
<br>
<a > <img src="Designer.jpeg" alt="Markdownify" width="200"> </a>
</h1>
<p align="center">
  <a href="#-inspiration">Inspiration</a> •
  <a href="#-data-collection">Data Collection</a> •
  <a href="#-requirements">Requirements</a> •
  <a href="#-features">Features</a> •
  <a href="#-project-files-structure">Project Structure</a> •
  <a href="#-license">License</a>
</p>

The **Handwriting Classification** project was conceived to identify and classify learning disabilities, known as Specific Learning Disorders (DSA), in preschool children through a series of specific tasks. These tasks include drawing, copying shapes, and writing sentences or numbers, each designed to reveal potential difficulties in various cognitive areas of the child.

The primary goal of this study is the early diagnosis of learning disabilities in children. By identifying these issues promptly, appropriate therapies can be provided by psychologists and psychotherapists. Early and targeted intervention can significantly improve therapeutic outcomes, allowing these disabilities to be addressed before they become more ingrained and difficult to manage.

The project employs advanced machine learning and deep learning techniques to analyze children's handwriting patterns. These algorithms enable accurate and reliable classification of potential learning disabilities. Utilizing these technologies makes the project a valuable tool for specialists, providing concrete and timely support in identifying children's needs. 


## 🌟 Inspiration
The data collection technique that enabled us to engage preschool subjects was storytelling. Specifically, we used a charming narrative to capture the children's interest and cooperation:

*"This is the story of Bipi, a little Martian who has come to Earth to explore our world, and Hanumi, the monkey who will help him learn about the things humans do. Bipi wants to learn what people do on planet Earth. For example, he wants to go to school like human children to learn how to draw and write. Can you help him see how it's done?"*

Through this engaging story, children were motivated to perform the tasks assigned to them. The narrative not only captivated their imagination but also provided a context in which they could relate to the activities. This approach proved effective in making the children feel comfortable and enthusiastic about participating in the study.

By framing the tasks within the story, we were able to collect data in a natural and stress-free manner. The children, engrossed in helping Bipi and Hanumi, approached the drawing, shape copying, and writing tasks with genuine interest and effort. This method ensured that the data collected was reflective of their true abilities, without the pressure often associated with formal testing environments.

The use of storytelling as a data collection method highlights the importance of creating a child-friendly and engaging atmosphere. It acknowledges that preschool children respond positively to imaginative and interactive scenarios, which can significantly enhance their willingness to participate and the quality of the data obtained. This innovative approach not only facilitated the data collection process but also underscored the potential of using creative techniques in educational and psychological research.

## 💾 Data Collection
The collected dataset comes from various subjects who were asked to complete 21 tasks. Each subject was provided with a graphic tablet and a small monitor to perform the tasks, with the writing unit (the pen) potentially varying during the task. The collected data is in the form of time series, and each completed task has an associated .CSV file. Each task contains the following features:

- `Timestamp`: The exact date and time when the data was recorded, following the ISO 8601 format with millisecond precision.
- `PointX`: The X coordinate of the pen tip on the tablet surface, representing the horizontal position.
- `PointY`: The Y coordinate of the pen tip on the tablet surface, representing the vertical position.
- `Phase`: Indicates the interaction state of the pen, such as 'Hover' (suspended), 'Touch' (touch), or other states.
- `Pressure`: The amount of pressure applied by the pen.
- `PointDisplayX`: The X coordinate on the display where the pen interaction is mapped. This is the translated position on the screen.
- `PointDisplayY`: The Y coordinate on the display where the pen interaction is mapped. This is the translated position on the screen.
- `PointRawX`: The raw value of the X coordinate from the pen sensor.
- `PointRawY`: The raw value of the Y coordinate from the pen sensor.
- `PressureRaw`: The raw value of the pressure from the pen sensor.
- `TimestampRaw`: The raw timestamp value, in a less readable format, representing a low-level timing mechanism.
- `Sequence`: A sequential number representing the order of data points, useful for reconstructing the interaction sequence.
- `Rotation`: The rotation angle of the pen in degrees.
- `Azimuthv`: The angle between the pen and the vertical axis (i.e., the tilt angle).
- `Altitude`: The angle between the pen and the horizontal plane.
- `TiltX`: The tilt of the pen along the X axis, indicating the direction and degree of tilt.
- `iltY`: The tilt of the pen along the Y axis, indicating the direction and degree of tilt.
- `PenId`: The identifier of the pen, useful when multiple pens are used.

> [!CAUTION]
> The data collected will be processed in accordance with the scientific and professional code of ethics, ensuring the utmost respect for the privacy of minor subjects and their vulnerabilities. This information will be used with the aim of improving the medical care provided and enabling early diagnosis of diseases in new patients. The collection and use of data will be done in full compliance with current regulations and with special attention to the protection of sensitive data, ensuring that every step of the process is transparent and secure for all involved.


## 🔩 Requirements

The project is based on `Python` *at version* `3.12.1` - one of the latest versions of Python at the time of writing **June 24'**. A few considerations:

- It is recommended to use a virtual environment to manage the dependencies of the project. For example, [conda](https://docs.conda.io/en/latest/).
- The requirements are listed in the `requirements.txt` file and can be installed using `pip install -r requirements.txt`.

This project leverages the following libraries:

- `addict` with version `2.4.0` - A Python library for easily managing nested dictionaries.
- `comet-ml` with version `3.43.2` - A platform for managing machine learning experiments and model tracking.
- `matplotlib` with version `3.8.4` - A comprehensive library for creating static, animated, and interactive visualizations in Python.
- `numpy` with version `1.26.4` - A fundamental package for scientific computing with Python, providing support for arrays and matrices, along with a collection of mathematical functions.
- `pandas` with version `2.2.2` - An open-source data analysis and manipulation tool built on top of Python.
- `pytorch` with version `2.3.0` - An open-source machine learning library based on the Torch library, utilized for applications including computer vision and natural language processing.
- `scikit-learn` with version `1.3.1` - A machine learning library for Python providing simple and efficient tools for data mining and data analysis.
- `seaborn` with version `0.13.2` - A Python data visualization library based on Matplotlib, offering a high-level interface for drawing attractive and informative statistical graphics.
- `torch` with version `0.10` - A library for machine learning that is a core component of the PyTorch ecosystem.
- `torchvision` with version `0.18.0` - A library that provides datasets, model architectures, and image transformations for computer vision tasks.
- `tqdm` with version `4.66.4` - A fast and extensible progress bar for Python and command-line interfaces.
- `yaml` with version `0.2.5` - A Python library for parsing and writing YAML, a human-friendly data serialization standard.

## 🚀 Features
In this section, there will be a brief description of the classes, detailing their functionality, purpose, and any implemented features. The section will be regularly updated with the latest functionalities.

- The **`dataset_management.py`** module implements a comprehensive dataset management system designed for deep learning applications. It features the `HWDataset_DL` class, which is responsible for loading, preprocessing, and augmenting datasets. This class handles various tasks including dataset balancing through Gaussian noise application to ensure class representation, and padding or truncating data to meet specified length requirements. Additionally, the module provides functionality for creating machine learning datasets by preprocessing raw data and calculating statistical features, enabling efficient and effective data handling for both training and evaluation phases.

- The **`model.py`** module serves as a comprehensive toolkit for developing both deep learning (DL) and machine learning (ML) architecture in Python. Within the DL section, it defines abstract base classes and concrete implementations for RNNs, LSTMs, and GRUs, offering configurable architectures for sequence processing tasks. These classes allow users to specify input sizes, hidden layer dimensions, output sizes, and dropout rates, facilitating flexibility in model design. On the ML front, the file includes implementations of K-Nearest Neighbors (KNN) and Support Vector Machines (SVM), which support model training, prediction, and hyperparameter tuning through grid search. Each class inherits from abstract base classes that define common interfaces for training, predicting, and evaluating models, ensuring consistency and ease of use across different model types.

- The **`train.py`** script is designed for training and evaluating deep learning (DL) and machine learning (ML) models using PyTorch and scikit-learn. It begins by importing essential libraries and utilities such as `torch`, `pandas`, `sklearn`, and custom modules like `dataset_management` and `model_classes`. The script defines functions for training deep learning models (`define_and_run_RNN`, `define_and_run_LSTM`, `define_and_run_GRU`) and evaluating them over multiple epochs using training and validation datasets. These functions utilize configurable model architectures (RNN, LSTM, GRU) specified in a YAML configuration file (`config.yaml`). The training process includes setting up model parameters, optimizer (e.g., Adam), and learning rate scheduler (`StepLR`). During training, progress and metrics are logged, including average loss and validation accuracy. Additionally, the script provides utilities for logging experiment metrics and visualizing results using tools like `matplotlib`.

- The **`test.py`** script serves as a comprehensive testing framework for evaluating Deep Learning models, specifically focusing on RNN architectures. It utilizes PyTorch for model creation and evaluation, leveraging components such as DataLoader for efficient data handling. The script imports necessary modules like `torch`, `torch.nn`, and `torch.optim` for model configuration and training. It integrates key functionalities including loading model configurations from YAML files, initializing model architectures (e.g., LSTM, GRU) with predefined parameters, and loading pretrained weights from saved model checkpoints. Additionally, the script computes evaluation metrics such as classification reports and confusion matrices, which are subsequently saved to designated directories.

- The **`utils.py**`** module provides essential utility functions for managing machine learning and deep learning workflows. It includes functionality for loading configurations from YAML files and saving visualizations of confusion matrices and classification reports as heatmap images. Additionally, the module offers tools for plotting training and validation loss over epochs. The utility functions also encompass printing confusion matrices and classification reports in a readable format. Furthermore, the module features methods to compute and print various classification metrics, such as accuracy, precision, recall, and F1-score, along with generating a detailed classification report. An evaluation function for neural network models is also implemented, which computes evaluation metrics on a validation set and returns them in a structured format.

##  📁 Project structure
The project is structured as follows:
```
hw_project/
│
├── config/
│   └──config_RNN.yaml
│
├── data_classes/ 
│   └──dataset_management.py
│
├── model_classes/ 
│   └──model.py
│
├── models/
│   ├──RNN_model.hwp
│   ├──LSTM_model.hwp
│   └──GRU_model.hwp
│
├── train_log/
│   ├──RNN/
│   │  ├──loss_comparison/
│   │  └──train_metrics/
│   │
│   ├──LSTM/
│   │  ├──loss_comparison/ 
│   │  └──train_metrics/
│   │
│   └──GRU/
│      ├──loss_comparison/
│      └──train_metrics/
│   
├── test_log/
│   ├──ML/
│   │  ├──KNN/
│   │  └──SVM/
│   │
│   └──DL/
│      ├──RNN/
│      ├──LSTM/
│      └──GRU/
│   
├── train.py
├── test.py
├── utils.py
├── requirements.txt
├── prepare.sh
├── README.md
├── LICENSE.txt
└── Designer.jpeg
```

- `config/`: Contains configuration files for various models.
  - `config_RNN.yaml`: YAML configuration file specifically tailored for the RNN model.
- `data_classes/`: Contains modules for dataset management.
  - `dataset_management.py`: Python script for dataset loading, preprocessing, and management.
- `model_classes/`: Encompasses modules related to model architecture and design.
  - `model.py`: Python script defining the architecture and operations of neural network models.
- `models/`: Stores serialized model files post-training.
  - `RNN_model.hwp`, `LSTM_model.hwp`, `GRU_model.hwp`: Serialized model files for RNN, LSTM, and GRU models, respectively.
- `train_log/`: Includes training logs and metrics for different models.
  - `RNN/`, `LSTM/`, `GRU/`: Directories dedicated to each model type.
    - `loss_comparison/`: Directory storing loss curves or comparisons during training.
    - `train_metrics/`: Directory for training metrics such as accuracy, loss, etc.
- `test_log/`: Stores testing logs and metrics for different models.
  - `ML/`, `DL/`: Main categories for Machine Learning and Deep Learning models.
    - `KNN/`, `SVM/`: Subdirectories under ML for specific ML model logs.
    - `RNN/`, `LSTM/`, `GRU/`: Subdirectories under DL for respective deep learning model logs.
- `train.py`: Script responsible for model training.
- `test.py`: Script for evaluating trained models.
- `utils.py`: Contains utility functions used throughout the project.
- `requirements.txt`: Lists dependencies required for the project.
- `prepare.sh`: Script used to set up the environment by installing project dependencies.
- `README.md`: Provides primary project documentation, including an overview of the project structure, setup instructions, and usage guidelines.
- `LICENSE.txt`: Contains licensing information governing the project.
- `Designer.jpeg`: An image file, potentially depicting a visual aspect relevant to the project.

## 🧾 License
Licensed under the terms of the CC BY-NC-SA 4.0, Copyright ©️ 2024 - present Andrea Pietro Arena (https://github.com/ArexJrr). You can see the full license and terms and conditions in the LICENSE file.