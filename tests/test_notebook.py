from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_valid():
    notebook = nbformat.read(ROOT / 'boston_housing_pipeline.ipynb', as_version=4)
    nbformat.validate(notebook)
    sources = '\n'.join(cell.source for cell in notebook.cells)
    assert 'GridSearchCV' in sources
    assert 'RandomForestRegressor' in sources
    assert 'train_test_split' in sources
