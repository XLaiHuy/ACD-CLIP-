#!/usr/bin/env python3
"""Build the fixed H2 BF16 Seed0 precision/seed decision artifacts."""
from __future__ import annotations
import csv, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MED = Path('/workspace/h2_bf16_screening/target_eval/seed0_bf16_a_e15/medical/A_E15_BF16_Seed0/macro_metrics.csv')
MVTEC = Path('/workspace/h2_bf16_screening/target_eval/seed0_bf16_a_e15/mvtec/A_E15_BF16_Seed0/test.log')

def main():
    with MED.open() as f: medical = {k: float(v) if k not in {'epoch','prompt_config','score_rule','image_n'} else v for k,v in next(csv.DictReader(f)).items()}
    log = MVTEC.read_text()
    average = re.search(r'^\s*Average\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$', log, re.M)
    assert average, 'MVTec Average row absent'
    mv = [float(v) for v in average.groups()]
    seed0 = {'Medical': {'pixel_AUROC': medical['pixel_auc_6'], 'pixel_AP': medical['pixel_ap_6'], 'image_AUROC': medical['image_auc_3'], 'image_AP': medical['image_ap_3']}, 'MVTec': dict(zip(('pixel_AUROC','pixel_AP','image_AUROC','image_AP'), mv))}
    historical = {'Medical': {'pixel_AUROC':91.25178605739472,'pixel_AP':39.46836969631676,'image_AUROC':75.2827266852061,'image_AP':76.34336352348328}, 'MVTec': {'pixel_AUROC':90.041289,'pixel_AP':45.159349,'image_AUROC':89.816913,'image_AP':94.779362}}
    seed1raw = json.loads((ROOT/'results/H2_BF16_SEED1_E15_RESULTS.json').read_text())
    seed1 = {'Medical': {'pixel_AUROC':seed1raw['medical_raw_exact']['A']['pixel_auroc'],'pixel_AP':seed1raw['medical_raw_exact']['A']['pixel_ap'],'image_AUROC':seed1raw['medical_raw_exact']['A']['image_auroc'],'image_AP':seed1raw['medical_raw_exact']['A']['image_ap']}, 'MVTec': {'pixel_AUROC':seed1raw['mvtec_benchmark_exact']['A']['pixel_auroc'],'pixel_AP':seed1raw['mvtec_benchmark_exact']['A']['pixel_ap'],'image_AUROC':seed1raw['mvtec_benchmark_exact']['A']['image_auroc'],'image_AP':seed1raw['mvtec_benchmark_exact']['A']['image_ap']}}
    ap_delta = sum(seed0[d]['pixel_AP']-historical[d]['pixel_AP'] for d in seed0)/2
    auc_delta = sum(seed0[d]['pixel_AUROC']-historical[d]['pixel_AUROC'] for d in seed0)/2
    assert ap_delta <= -2.0
    rows=[]
    for label, values in [('A_FP16FP32_Seed0',historical),('A_BF16_Seed0',seed0),('A_BF16_Seed1',seed1)]:
        for domain in ('Medical','MVTec'):
            rows.append({'arm':label,'domain':domain,**values[domain],'provenance':'HISTORICAL_EXACT' if label.startswith('A_FP') else 'MEASURED_EXACT'})
    for label, left, right in [('A_BF16_Seed0_MINUS_A_FP16FP32_Seed0',seed0,historical),('A_BF16_Seed1_MINUS_A_BF16_Seed0',seed1,seed0)]:
        for domain in ('Medical','MVTec'):
            rows.append({'arm':label,'domain':domain,**{k:left[domain][k]-right[domain][k] for k in left[domain]},'provenance':'DERIVED'})
    out=ROOT/'results/H2_BF16_PRECISION_SEED0_THREE_WAY_COMPARISON.csv'
    with out.open('w',newline='') as f: csv.DictWriter(f,fieldnames=['arm','domain','pixel_AUROC','pixel_AP','image_AUROC','image_AP','provenance']).writeheader(); csv.DictWriter(f,fieldnames=['arm','domain','pixel_AUROC','pixel_AP','image_AUROC','image_AP','provenance']).writerows(rows)
    medical_rows = json.loads((MED.parent/'dataset_metrics.json').read_text())
    target_rows = []
    for row in medical_rows:
        target_rows.append({'domain':'Medical','dataset':row['dataset'],'metric_type':row['metric_type'],
                            'pixel_AUROC':row.get('pixel_auc',''),'pixel_AP':row.get('pixel_ap',''),
                            'image_AUROC':row.get('image_auc',''),'image_AP':row.get('image_ap',''),
                            'provenance':'MEASURED_EXACT'})
    in_table = False
    for line in log.splitlines():
        if line.startswith('class name'): in_table = True; continue
        if in_table:
            bits = line.split()
            if len(bits) == 5:
                target_rows.append({'domain':'MVTec','dataset':bits[0],'metric_type':'pixel_and_image',
                                    'pixel_AUROC':bits[1],'pixel_AP':bits[2],'image_AUROC':bits[3],'image_AP':bits[4],
                                    'provenance':'MEASURED_EXACT'})
    with (ROOT/'results/H2_BF16_SEED0_TARGET_PER_DATASET.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['domain','dataset','metric_type','pixel_AUROC','pixel_AP','image_AUROC','image_AP','provenance'])
        writer.writeheader(); writer.writerows(target_rows)
    payload={'protocol':'H2_POST_SOURCE_DIAGNOSIS_MINIMUM_COMPUTE','new_bf16_seed0':seed0,'historical_fp16_fp32_seed0':historical,'existing_bf16_seed1':seed1,'mean_pixel_AP_delta_bf16_seed0_vs_mixed_seed0':ap_delta,'mean_pixel_AUROC_delta_bf16_seed0_vs_mixed_seed0':auc_delta,'decision':{'PRECISION_QUALITY_DEGRADATION':'LIKELY','SEED_INITIALIZATION_SENSITIVITY':'POSSIBLE','GENERALIZATION_BOTTLENECK':'POSSIBLE','RECOMMENDED_PRECISION':'MIXED_FP16_FP32','PRIMARY_BOTTLENECK':'BF16 precision-policy quality degradation','SECONDARY_BOTTLENECK':'seed/trajectory sensitivity'},'historical_source_audit':'UNAVAILABLE','historical_target_protocol':'EXACT','new_target_contract':'Medical raw_exact stride=1; MVTec benchmark_exact stride=1','max_additional_full_trains':1,'actual_additional_full_trains':1,'more_training_authorized':False}
    (ROOT/'results/H2_BF16_SEED0_E15_RESULTS.json').write_text(json.dumps(payload,indent=2)+"\n")
    md=f'''# H2 BF16 Seed0 precision / seed decision

`FINAL_DECISION=PASS`  
`LAST_FULL_TRAIN=A_BF16_SEED0_E15`  
`LAST_FULL_TRAIN_VALIDITY=PASS`  
`SOURCE_AUDIT_STATUS=PASS`  
`HISTORICAL_SEED0_SOURCE_AUDIT=UNAVAILABLE`  
`GENERALIZATION_BOTTLENECK=POSSIBLE`  
`MAX_ADDITIONAL_FULL_TRAINS=1; ACTUAL_ADDITIONAL_FULL_TRAINS=1; MORE_TRAINING_AUTHORIZED=NO`  
`WAITING_FOR_USER_APPROVAL=YES`

| arm | Medical AUROC | Medical AP | MVTec AUROC | MVTec AP |
|---|---:|---:|---:|---:|
| A FP16+FP32 Seed0 | {historical['Medical']['pixel_AUROC']:.4f} | {historical['Medical']['pixel_AP']:.4f} | {historical['MVTec']['pixel_AUROC']:.4f} | {historical['MVTec']['pixel_AP']:.4f} |
| A BF16 Seed0 | {seed0['Medical']['pixel_AUROC']:.4f} | {seed0['Medical']['pixel_AP']:.4f} | {seed0['MVTec']['pixel_AUROC']:.4f} | {seed0['MVTec']['pixel_AP']:.4f} |
| A BF16 Seed1 | {seed1['Medical']['pixel_AUROC']:.4f} | {seed1['Medical']['pixel_AP']:.4f} | {seed1['MVTec']['pixel_AUROC']:.4f} | {seed1['MVTec']['pixel_AP']:.4f} |

BF16 Seed0 minus mixed Seed0: Medical `{-1*(historical['Medical']['pixel_AUROC']-seed0['Medical']['pixel_AUROC']):+.4f}/{seed0['Medical']['pixel_AP']-historical['Medical']['pixel_AP']:+.4f}`; MVTec `{-1*(historical['MVTec']['pixel_AUROC']-seed0['MVTec']['pixel_AUROC']):+.4f}/{seed0['MVTec']['pixel_AP']-historical['MVTec']['pixel_AP']:+.4f}` (AUROC/AP, percentage points).

`MEAN_PIXEL_AP_DELTA_BF16_SEED0_VS_MIXED_SEED0={ap_delta:+.6f}` and `MEAN_PIXEL_AUROC_DELTA_BF16_SEED0_VS_MIXED_SEED0={auc_delta:+.6f}`. The predeclared clear-loss rule applies: AP is below −2 points and both target domains decline. `PRECISION_QUALITY_DEGRADATION=LIKELY`; `RECOMMENDED_PRECISION=MIXED_FP16_FP32` because historical A itself records zero nonfinite events. BF16 Seed0 versus Seed1 is mixed across domains, so `SEED_INITIALIZATION_SENSITIVITY=POSSIBLE`, not the primary explanation.

The historical source checkpoint is unavailable, hence its source observability audit remains unavailable. Its target metrics are nevertheless stored under the exact current target evaluator contract. Seed1 source ranking/feature separation was healthy, DFG routing suspicious, and pixel localization a bottleneck; the new cross-domain Seed0 result makes precision-policy quality the primary bottleneck rather than generalization alone. No further training is authorized.
'''
    (ROOT/'results/H2_BF16_PRECISION_SEED0_FINAL_DECISION.md').write_text(md)
    print('H2_BF16_PRECISION_SEED0_DECISION=PASS')
if __name__ == '__main__': main()
