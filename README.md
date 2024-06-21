# Handwriting ✍ Classification
<h1 align="center">
<br>
<a > <img src="Designer.jpeg" alt="Markdownify" width="200"> </a>
</h1>
<p align="center">
  <a href="#-inspiration">Getting Started</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#download">Download</a> •
  <a href="#credits">Credits</a> •
  <a href="#related">Related</a> •
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

This project is based on the following libraries:

- `matplotlib` with version `3.8.4` - A comprehensive library for creating static, animated, and interactive visualizations in Python.
- `numpy` with version `1.26.4` - A fundamental package for scientific computing with Python, providing support for arrays and matrices, along with a collection of mathematical functions to operate on these data structures.
- `pandas` with version `2.2.2` - An open-source data analysis and manipulation tool, built on top of the Python programming language.
- `pytorch` with version `2.3.0` - An open-source machine learning library based on the Torch library, used for applications such as computer vision and natural language processing.
- `scikit-learn` with version `1.3.1` - A machine learning library for Python, providing simple and efficient tools for data mining and data analysis.
- `scipy` with version `1.13.0` - A Python library used for scientific and technical computing, building on the NumPy array object.
- `seaborn` with version `0.13.2` - A Python data visualization library based on Matplotlib, providing a high-level interface for drawing attractive and informative statistical graphics.
- `torch` with version `0.10` - A library for machine learning that is a core part of the PyTorch ecosystem.
- `torchvision` with version `` - A library that provides datasets, model architectures, and image transformations for computer vision.
- `tqdm` with version `4.66.4` - A fast, extensible progress bar for Python and CLI.
- `yaml` with version `0.2.5` - A Python library for parsing and writing YAML, a human-friendly data serialization standard.

## 🚀 Features
Describe models used ML and DL 




##  📁 Project files structure
The project is structured as follows:
```
hw_project/
│
├── config/
│   ├──config_RNN.yaml
│
├── data_classes/ 
│   ├──cds_.py
│   └──svm_ds.py
│
├── model_classes/ 
│   ├──model.py
│   └──model_svm.py
│
├── models/
│   ├──model.py
│   └──model_svm.py
│
│
├── train.py
├── test.py
├── requirements.txt
├── prepare.sh
├── README.md
├── LICENSE
│


└── ...
```

continue to decribe classes





## 🧾 License
Licensed under the terms of the CC BY-NC-SA 4.0, Copyright ©️ 2024 - present Andrea Pietro Arena (https://github.com/ArexJrr). You can see the full license and terms and conditions in the LICENSE file.