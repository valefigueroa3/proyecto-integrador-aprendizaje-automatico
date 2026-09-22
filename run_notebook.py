from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient


relative_path = sys.argv[1] if len(sys.argv) > 1 else "notebooks/01_carga_y_auditoria.ipynb"
notebook_path = Path(relative_path).resolve()
notebook = nbformat.read(notebook_path, as_version=4)
client = NotebookClient(
    notebook,
    timeout=600,
    kernel_name="python3",
    resources={"metadata": {"path": str(notebook_path.parent)}},
)
client.execute()
nbformat.write(notebook, notebook_path)
print(f"Notebook ejecutado: {notebook_path}")
