import copy, math
import torch
from data.text import random_batch, sequential_batches
from photonic.calibration import CalibrationCollector

@torch.no_grad()
def evaluate(model,data,batch_size,block_size,device,max_batches,engine=None,calibration=None,photonic_lm_head=True):
    model.eval(); losses=[]
    for x,y in sequential_batches(data,batch_size,block_size,device,max_batches):
        _,loss=model(x,y,engine=engine,calibration=calibration,photonic_lm_head=photonic_lm_head)
        losses.append(float(loss.detach().cpu()))
    loss=sum(losses)/len(losses); return loss,math.exp(min(loss,20.0))

def train_digital(model,train_data,val_data,cfg,device):
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],betas=(.9,.95))
    best=float('inf'); best_state=None
    for step in range(1,cfg['train_steps']+1):
        model.train(); x,y=random_batch(train_data,cfg['batch_size'],cfg['block_size'],device)
        _,loss=model(x,y); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step==1 or step%cfg['eval_interval']==0 or step==cfg['train_steps']:
            val,_=evaluate(model,val_data,cfg['batch_size'],cfg['block_size'],device,cfg['eval_batches'])
            print(f'step={step:5d} train={float(loss.detach().cpu()):.4f} val={val:.4f}')
            if val<best:
                best=val; best_state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
    if best_state is not None: model.load_state_dict(best_state)
    model.to(device); return best

@torch.no_grad()
def calibrate(model,data,cfg,device,percentile):
    col=CalibrationCollector(); model.eval()
    for x,_ in sequential_batches(data,cfg['batch_size'],cfg['block_size'],device,cfg['calibration_batches']): model(x,collector=col)
    return col.finalize(percentile)
