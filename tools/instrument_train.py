import re
import sys

def instrument():
    with open("train_profile.py", "r") as f:
        content = f.read()

    # Add imports and timer setup
    header = """
import time
import torch
from collections import defaultdict
import json

class Timer:
    def __init__(self):
        self.starts = {}
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)
        self.warmup = 20
        self.active = False
        
    def start(self, name):
        if not self.active: return
        torch.cuda.synchronize()
        self.starts[name] = time.perf_counter()
        
    def end(self, name):
        if not self.active: return
        torch.cuda.synchronize()
        if name in self.starts:
            self.totals[name] += time.perf_counter() - self.starts[name]
            self.counts[name] += 1

global_timer = Timer()
"""
    content = content.replace("import torch\n", "import torch\n" + header, 1)

    # Instrument Data Loading
    content = content.replace("for batch_idx, input_data in enumerate(progress, start=1):", 
                              "global_timer.start('data_loading')\n        for batch_idx, input_data in enumerate(progress, start=1):")
    
    content = content.replace("image = input_data[\"image\"].to(device, non_blocking=args.pin_memory)",
                              "global_timer.end('data_loading')\n            global_timer.start('host_to_device')\n            image = input_data[\"image\"].to(device, non_blocking=args.pin_memory)")
    
    content = content.replace("with _phase4_autocast(device, args.precision):",
                              "global_timer.end('host_to_device')\n            with _phase4_autocast(device, args.precision):")
                              
    content = content.replace("visual_output = model(image, return_phase4_features=True)",
                              "global_timer.start('visual_forward')\n                visual_output = model(image, return_phase4_features=True)\n                global_timer.end('visual_forward')")
    
    content = content.replace("h6_batch = model.h6.build_batch(",
                              "global_timer.start('h6_forward')\n                h6_batch = model.h6.build_batch(")
    content = content.replace("hybrid_alpha=hybrid_alpha\n                )",
                              "hybrid_alpha=hybrid_alpha\n                )\n                global_timer.end('h6_forward')")
                              
    content = content.replace("cls_loss = F.cross_entropy(cls_pred.float(), label)",
                              "global_timer.start('loss_construction')\n                cls_loss = F.cross_entropy(cls_pred.float(), label)")
                              
    content = content.replace("drift_gradient_report = h6_drift_gradient_attribution(",
                              "global_timer.end('loss_construction')\n                global_timer.start('expensive_diagnostics')\n                drift_gradient_report = h6_drift_gradient_attribution(")
    content = content.replace("h6_drift_parameter_groups(model),\n                    )",
                              "h6_drift_parameter_groups(model),\n                    )\n                global_timer.end('expensive_diagnostics')")

    # If it didn't end loss construction due to drift diag missing
    content = content.replace("if not torch.isfinite(total_loss).all():",
                              "global_timer.end('loss_construction')\n            if not torch.isfinite(total_loss).all():")
                              
    content = content.replace("scaler.scale(total_loss / args.grad_accum_steps).backward()",
                              "global_timer.start('backward')\n            scaler.scale(total_loss / args.grad_accum_steps).backward()\n            global_timer.end('backward')")

    content = content.replace("do_step = batch_idx % args.grad_accum_steps == 0 or batch_idx == epoch_batch_limit",
                              "global_timer.start('optimizer_step')\n            do_step = batch_idx % args.grad_accum_steps == 0 or batch_idx == epoch_batch_limit")
                              
    content = content.replace("if args.h6_drift_diagnostics and \"batch_000_after_first_optimizer_step\" not in drift_snapshots:",
                              "global_timer.end('optimizer_step')\n                if args.h6_drift_diagnostics and \"batch_000_after_first_optimizer_step\" not in drift_snapshots:")

    # Add loop control for warmup and measurement
    content = content.replace("for batch_idx, input_data in enumerate(progress, start=1):",
                              """for batch_idx, input_data in enumerate(progress, start=1):
            if batch_idx > 120: 
                print("Profile finished")
                res = {k: v/global_timer.counts[k] for k, v in global_timer.totals.items()}
                with open("runs/phase4/p1_fast_audit/runtime_profile_before.json", "w") as jf:
                    json.dump(res, jf, indent=2)
                sys.exit(0)
            if batch_idx > 20: global_timer.active = True""")
            
    # Restart data loading timer at end of loop iteration
    content = content.replace("global_timer.start('data_loading')\n        for", "for")
    content = content.replace("progress.update(1)", "progress.update(1)\n            global_timer.start('data_loading')")

    with open("train_profile.py", "w") as f:
        f.write(content)

if __name__ == "__main__":
    instrument()
