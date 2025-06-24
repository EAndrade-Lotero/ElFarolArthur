from pathlib import Path

PATHS = {
    'images': Path(Path.cwd(), '..', 'images').resolve(),
    'folder_figures_for_paper': Path(Path.cwd(), '..', 'images', 'Figures for paper').resolve(),
    'simulated_data': Path(Path.cwd(), '..', 'data').resolve(),
}

# Chech if the paths exist
for name, folder in PATHS.items():
    folder.mkdir(parents=True, exist_ok=True)