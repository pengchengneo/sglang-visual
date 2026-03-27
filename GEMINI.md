# SGLang Visualizer

## Project Overview

SGLang Visualizer is an interactive web-based tool for visualizing SGLang inference optimizations. It allows users to adjust parameters such as Tensor Parallelism (TP), quantization, scheduling strategies, speculative decoding, and CUDA Graphs, and observe their real-time effects on the model architecture, runtime topology, and GPU memory distribution.

The application is divided into two main views:
1.  **Compute Plane**: Visualizes the internal structure of the model, including operator weight shapes, TP partitioning strategies, communication operators, and a detailed GPU memory panel.
2.  **Control Plane**: Visualizes the complete SGLang runtime architecture, showing the data flow from HTTP API to GPU, including TokenizerManager, Scheduler, Worker Pool, and ModelRunner.

## Architecture & Technologies

The project is a full-stack application structured into two main components:

### Frontend (`frontend/`)
An interactive visualization web application.
*   **Technologies**: React 19, TypeScript 5.9, Vite 7
*   **State Management**: Native React `useState` and `useMemo` (no external state management library).
*   **Key Directories**:
    *   `src/components/`: Contains UI components grouped by feature (e.g., `sidebar`, `gpu`, `pipeline`, `controlplane`).
    *   `src/utils/`: Math and layout utilities for computing memory, TP shapes, and Sankey diagram connections.
    *   `public/presets/`: Pre-generated model JSON data.

### Backend (`backend/`)
A Python-based CLI tool used to analyze model structures and generate the JSON data required by the frontend.
*   **Technologies**: Python 3.10+, Pydantic 2, `huggingface_hub`
*   **Functionality**: Exposes a CLI command `sglang-tp-viz` to fetch model configurations from HuggingFace or local paths and output architecture JSON files.

## Building and Running

### Frontend

The frontend can run standalone using pre-generated data.

```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Run linting
npm run lint
```

### Backend (Optional)

The backend is mainly for generating new model preset JSONs.

```bash
# Navigate to the backend directory
cd backend

# Install the package with development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

### Generating Presets

To update or regenerate all preset model JSONs used by the frontend:

```bash
# From the project root
python generate_presets.py
```

## Development Conventions

*   **Frontend**: Follows standard React and TypeScript practices. Uses Vite as the build tool. Styling is likely handled via standard CSS (`App.css`, `index.css`).
*   **Backend**: Uses standard Python packaging (`pyproject.toml`) and `pytest` for testing. Models data using `pydantic`.
*   **Deployment**: The application is deployed to GitHub Pages via a GitHub Actions workflow (`.github/workflows/deploy.yml`) triggered by pushes to the `main` branch.