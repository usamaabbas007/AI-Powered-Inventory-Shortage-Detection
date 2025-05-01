
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
from model import InventoryShortageNet

app = FastAPI()

class InputData(BaseModel):
    stock_level: float
    daily_sales: float
    sales_velocity: float
    on_order_qty: float
    delivery_delay: float
    supplier_reliability: float

# Load model
input_dim = 6
model = InventoryShortageNet(input_dim)
model.load_state_dict(torch.load("models/shortage_model.pth"))
model.eval()

@app.post("/predict")
def predict(data: InputData):
    x = np.array([
        data.stock_level,
        data.daily_sales,
        data.sales_velocity,
        data.on_order_qty,
        data.delivery_delay,
        data.supplier_reliability
    ], dtype=np.float32).reshape(1, -1)
    x_tensor = torch.tensor(x)
    with torch.no_grad():
        output = model(x_tensor).item()
    return {
        "shortage_probability": round(output, 4),
        "shortage_predicted": output > 0.5
    }
