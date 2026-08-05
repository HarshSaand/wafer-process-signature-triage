import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wafer_process_ai.model import WaferFusionCNN, TemperatureScaler, cartesian_to_polar
from wafer_process_ai.metrics import expected_calibration_error, multiclass_brier_score, nearest_neighbors

def test_polar_shape_and_gradients():
    x=torch.rand(2,1,32,48,requires_grad=True); y=cartesian_to_polar(x,16,20)
    assert y.shape==(2,1,16,20); y.mean().backward(); assert x.grad is not None

def test_fusion_model_contract():
    m=WaferFusionCNN(9,num_features=4,width=4,embedding_dim=8)
    x=torch.rand(3,1,32,32); f=torch.rand(3,4)
    assert m(x,f).shape==(3,9); assert m.embed(x,f).shape==(3,128)
    with pytest.raises(ValueError): m(x)

def test_temperature_positive_and_metrics_finite():
    s=TemperatureScaler(); logits=torch.tensor([[3.,0.],[0.,3.]]); y=torch.tensor([0,1]); s.fit(logits,y,max_iter=5)
    assert 0.05 <= s.temperature.item() <= 20
    p=np.array([[.8,.2],[.1,.9]])
    assert 0 <= expected_calibration_error(y.numpy(),p) <= 1
    assert 0 <= multiclass_brier_score(y.numpy(),p) <= 2

def test_retrieval_order():
    got=nearest_neighbors([1,0],[[0,1],[1,0],[.8,.2]],["a","b","c"],2)
    assert [x["label"] for x in got]==["b","c"]
