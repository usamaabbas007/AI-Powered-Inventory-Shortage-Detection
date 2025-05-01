AI-Powered Inventory Shortage Detection
Forecasting retail stockouts with neural networks, dbt-style pipelines, and synthetic data

Overview
This project simulates a real-world retail supply chain (inspired by companies like H-E-B) and uses a neural network to detect and predict inventory shortages (stockouts). It demonstrates the end-to-end AI workflow — from raw multi-location data generation to modeling, deployment, and dashboarding.

Designed for AI Engineering roles, it showcases: - Clean data pipelines (simulated Snowflake + dbt) - Feature-rich synthetic dataset - PyTorch-based neural network classifier - FastAPI inference server - Streamlit dashboard for stakeholders - Full modular structure — easy to extend or deploy

Features
5 store locations × 100 products × 365 days
Labeled shortage events (shortage_event = 1 if stockout occurred)
Sales trends, supplier delays, and delivery patterns
Fully scriptable and reproducible with synthetic data
Project Structure
├── data/
│   ├── raw/              ← Inventory, sales, delivery data (CSV)
│   └── features/         ← Engineered features (via dbt logic)
├── dbt/
│   ├── models/           ← SQL-style transformations
│   ├── seeds/            ← Supplier reliability data
├── models/               ← Trained neural network weights
├── src/
│   ├── train_model.py    ← Trains the neural network
│   ├── inference_api.py  ← FastAPI server for predictions
│   ├── dashboard_app.py  ← Streamlit visualization
│   └── utils.py / model.py
├── requirements.txt
├── README.md
How to Run Locally
Install requirements
pip install -r requirements.txt
Train the model
python src/train_model.py
Launch FastAPI
uvicorn src.inference_api:app --reload
Launch Dashboard
streamlit run src/dashboard_app.py
Model Details
Feedforward neural network
6 input features (stock level, sales, velocity, delivery delay, etc.)
Binary output: shortage_event
Trained with BCE loss and dropout regularization
Use Case
This is ideal for: - Retail companies managing multi-location inventory - Preventing lost sales due to out-of-stock issues - Automating alerts based on live demand + supplier reliability

License
MIT License

Credits
Built by Raja Usama Abbas – AI Engineer
GitHub: usamaabbas007
