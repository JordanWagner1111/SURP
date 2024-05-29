# Where do stars explode in the ISM? -- The distribution of dense gas around massive stars and supernova remnants in M33 Repository

## Overview

This repository aims to provide clear and concise code for creating plots in Jupyter Notebooks, tailored to our research. The primary functionality is within the `data` folder, particularly in the `DataObject.py` and `Regions.py` files. These files define the classes and methods utilized in the notebooks located in the `notebooks` folder.

## Structure

- **data folder**:
  - **DataObject.py**: Contains class `DataObject` and it's methods for data manipulation and preprocessing.
  - **Regions.py**: Contains class `Region` which inherits `DataObject` methods for handling different regions within the data.

- **notebooks folder**:
  - Jupyter Notebooks that use the classes and methods from the `data` folder to generate plots and perform analyses.

## Usage

1. **Clone the repository into a folder**:
    ```bash
    git clone https://github.com/JordanWagner1111/SURP.git
    ```

2. **Navigate to the repository**:
    ```bash
    cd name-of-your-repo-folder
    ```

3. **Install required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Open Jupyter Notebooks**:
    ```bash
    jupyter notebook
    ```
   Navigate to the `notebooks` folder and open the desired notebook to start creating plots.

## Future Directions

One possible future upgrade for this repo is to allow it to become a template that can import different data sets. This will streamline the process of conducting similar research methods across various data sources, allowing efficiency and consistency for future research with the same methodology.

![gas_maps_with_stars](https://github.com/JordanWagner1111/SURP/assets/105239335/8de5378a-a641-467c-b564-d347731c9b09)
(Sarbadhicary et al 2023)
