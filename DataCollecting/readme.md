# GitHub Developer Metrics Collector

This project collects data from 100 small GitHub projects and computes various metrics for each developer in those projects.

## Description

The script `collect_data.py` selects 100 small GitHub repositories based on specific criteria and collects commit data from those repositories. It then computes a set of metrics for each developer based on their commit history. The metrics are saved in a CSV file `developer_metrics.csv` for further machine learning tasks or analysis.

## Metrics Computed

The following metrics are computed for each developer:

- **period**: Number of days between first and last commit
- **days**: Number of days with at least one commit
- **weeks**: Number of weeks with at least one commit
- **timediff**: Median of days between successive commits
- **commits**: Number of authored commits
- **loc per commit**: Median lines of code modified per commit
- **weekend**: Percentage of commits during the weekend
- **night**: Percentage of commits between midnight and 6 am
- **morning**: Percentage of commits between 6 am and noon
- **afternoon**: Percentage of commits between noon and 6 pm
- **evening**: Percentage of commits between 6 pm and midnight
- **office**: Percentage of commits between 8 am and 5 pm
- **most active hour**: Hour of day with the highest number of commits
- **beginning regular**: Hour of day when weekday activity starts
- **end regular**: Hour of day when weekday activity ends
- **length regular**: Length of weekday activity period

## Prerequisites

- Python 3.6+
- A GitHub personal access token with appropriate permissions to access repository data.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/github-developer-metrics.git
   cd github-developer-metrics

2. Install the required packages:

  ```bash

pip install -r requirements.txt
 ```
Create a file named token.txt in the project directory and paste your GitHub personal access token into it.

#### Note: Keep your token secure and do not share it.

## Usage

Run the script:

```bash
python collect_data.py
```
The script will output progress messages and save the developer metrics to developer_metrics.csv.
