import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from photonic.config import PhotonicConfig
from photonic.core import PhotonicLinearEngine, LayerCalibration


def main():
    torch.manual_seed(0)
    x=torch.randn(4,8)
    w=torch.randn(5,8)
    b=torch.randn(5)
    cal=LayerCalibration(input_abs_range=float(x.abs().max()))

    # Internal sanity check only: ideal optical MAC must reduce to digital linear.
    eng=PhotonicLinearEngine(PhotonicConfig(), 'ideal')
    y=eng.forward(x,w,b,'test',cal)
    yd=torch.nn.functional.linear(x,w,b)
    assert torch.allclose(y,yd,atol=1e-5,rtol=1e-5), (y-yd).abs().max().item()

    # Realistic mode must accept zero random-noise scaling.
    cfg=PhotonicConfig(noise_scale=0.0)
    eng2=PhotonicLinearEngine(cfg,'realistic')
    y2=eng2.forward(x,w,b,'test2',cal)
    assert torch.isfinite(y2).all()
    print('test_core: PASS')


if __name__=='__main__':
    main()
