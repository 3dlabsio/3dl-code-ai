# 3DL Code AI

Welcome to **3DL Code AI**, a project by **3D Labs (3DL)** designed to simplify code understanding and management. This tool provides an efficient and seamless way to ingest entire code repositories and index them into a vector database for AI-powered search, analysis, and retrieval.

---

## 🚀 Features

- **Devcontainer Setup**: Simplifies the development environment, ensuring consistency across setups.
- **Code Ingestion**: Scans an entire code repository, extracting meaningful data for indexing.
- **Vector Database Integration**: 
  - Initial support for **Deeplake**.
  - Planned expansion for additional vector databases.
- **Python-Powered**: Leveraging Python's robust ecosystem for flexibility and power.

---

## 📦 Getting Started

### Prerequisites

Ensure you have the following installed on your machine:
- Docker (for devcontainer support)
- Python 3.8+
- Git (for repository cloning)
- Optional: A configured vector database like **Deeplake**.

### Setup

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/3d-labs/3dl-code-ai.git
    cd 3dl-code-ai
    ```

2. **Open in Devcontainer**:
    - Use Visual Studio Code.
    - Install the **Dev Containers** extension.
    - Open the folder in a containerized environment:
        ```text
        Open Command Palette (Ctrl+Shift+P) -> Remote-Containers: Reopen in Container
        ```

3. **Install Dependencies**:
    The dependencies will be automatically installed by the container build.

4. **Configure Database**:
    - Update the `.env` file with your database credentials and configurations.

---

## 🛠️ Usage

### Ingest a Code Repository

1. Place the target repository in the working directory or provide its path.
2. Run the ingestion script:
    ```bash
    python codeai.py <repository_url>
    ```

### Indexing with Deeplake

Ensure your Deeplake instance is running and accessible. The script will automatically detect and index the repository data.

---

## 🌟 Roadmap

1. Expand vector database support:
    - Pinecone
    - Weaviate
    - Milvus
2. Add support for more programming languages and complex project structures.
3. Integrate AI-assisted insights and querying for indexed codebases.

---

## 🤝 Contributions

We welcome contributions to 3DL Code AI! To get started:
1. Fork the repository.
2. Create a feat
