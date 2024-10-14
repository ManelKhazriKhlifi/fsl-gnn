import time
import torch
from args import args

_stats = {}

def nvar(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        return x.item() if x.dim() == 0 else x.numpy()
    return x

def log(message, to_file=True):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_line = f"{message}"
    print(log_line, flush=True)
    if args.log_file and to_file:
        with open(args.log_file, 'a') as f:
            print(f"# {current_time} # {log_line}", flush=True, file=f)

def log_scalar(tag, value, global_step):
    _stats[tag] = (nvar(value), global_step)

def log_text(tag, text, global_step):
    _stats[tag] = (text, global_step)

def log_confusion_matrix(cm):
        # Determine the maximum width of the numbers in the confusion matrix for formatting
        max_width = max(len(str(x)) for row in cm for x in row) + 2  # Add some padding

        log("Confusion Matrix:")
        for line in cm:
            formatted_line ='  ' + ''.join(f"{num:>{max_width}}" for num in line)  # Right align numbers with fixed width
            log(formatted_line)
            
def log_step(global_step=None, max_step=None):
    console_out = f"epoch: {global_step}/{max_step if max_step else ''}"
    details = ' '.join(f"{k}: {v[0]:.4f}" if isinstance(v[0], float) else f"{k}: {v[0]}" for k, v in _stats.items())
    
    if details:
        log(f"{console_out} {details}")

    _stats.clear()