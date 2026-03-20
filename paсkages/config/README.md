# Config Module

The `config` module provides a centralized location for managing configuration settings for the apartment rent data project.

## `Config` Class

The `Config` class is the main entry point for accessing project-wide configuration values.

### Attributes

- `PROJECT_ROOT`: The absolute path to the project root directory.
- `DATA_DIR`: The absolute path to the data directory.

### Methods

- `get_data_path(filename)`: Returns the full path to a data file by joining the `DATA_DIR` with the provided `filename`.

## Usage

To use the `Config` class, simply import it and access the attributes or methods as needed:

```python
from paсkages.config.config.config import Config

data_path = Config.get_data_path('apartment_data.csv')
print(data_path)