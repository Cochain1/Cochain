# Cochain ➰

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![Neo4j Version](https://img.shields.io/badge/Neo4j-4.4.0-brightgreen)](https://neo4j.com/)

## Installation ⚙️

### Prerequisites
- Python 3.8+
- Neo4j 4.x ([Install Guide](https://neo4j.com/docs/operations-manual/current/installation/))

### Steps
1. Clone the repo:
   ```bash
   git clone https://github.com/Cochain1/Cochain.git
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Configure Neo4j credentials in config.yaml:
   ```bash
   database:
     uri: "bolt://localhost:7687"
     user: "neo4j"
     password: "your_password"
