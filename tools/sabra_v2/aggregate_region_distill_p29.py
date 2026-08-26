"""Aggregate all twelve P29 immutable held scores."""
from __future__ import annotations
import argparse,csv,json,statistics
from pathlib import Path
from tools.sabra.data import EXPECTED_VISA_CLASSES
from tools.sabra_v2.region_cache import atomic_write_json

def make_parser():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);return p

def run(args):
 rows=[]
 for name in EXPECTED_VISA_CLASSES:
  path=args.run_root/name/"metrics"/"p29_held_metrics.json"
  if not path.is_file():raise RuntimeError(f"all P29 folds must be scored before aggregation; missing {path}")
  m=json.loads(path.read_text());
  if m.get("held_class")!=name or m.get("fit_or_teacher_steps")!=0:raise RuntimeError(f"P29 metric provenance failed {name}")
  n,s=m["native_metrics"],m["p29_metrics"];rows.append({"class":name,"native_pAP":float(n["pAP"]),"p29_pAP":float(s["pAP"]),"delta_pAP":float(s["pAP"]-n["pAP"]),"native_pAUROC":float(n["pAUROC"]),"p29_pAUROC":float(s["pAUROC"]),"delta_pAUROC":float(s["pAUROC"]-n["pAUROC"])})
 deltas=[r["delta_pAP"] for r in rows];positive=sorted((v for v in deltas if v>0),reverse=True);total=sum(positive);result={"schema_version":"P29_AGGREGATE_V1","fold_count":12,"native_macro_pAP":statistics.fmean(r["native_pAP"] for r in rows),"p29_macro_pAP":statistics.fmean(r["p29_pAP"] for r in rows),"delta_macro_pAP":statistics.fmean(r["delta_pAP"] for r in rows),"native_macro_pAUROC":statistics.fmean(r["native_pAUROC"] for r in rows),"p29_macro_pAUROC":statistics.fmean(r["p29_pAUROC"] for r in rows),"delta_macro_pAUROC":statistics.fmean(r["delta_pAUROC"] for r in rows),"improving_class_count":sum(v>0 for v in deltas),"non_regressing_class_count":sum(v>=0 for v in deltas),"regressing_class_count":sum(v<0 for v in deltas),"median_delta_pAP":statistics.median(deltas),"best_delta_pAP":max(deltas),"worst_delta_pAP":min(deltas),"top_1_positive_gain_concentration":positive[0]/total if total else 0.0,"top_2_positive_gain_concentration":sum(positive[:2])/total if total else 0.0,"classes":rows};args.output.mkdir(parents=True,exist_ok=True);atomic_write_json(args.output/"P29_AGGREGATE.json",result)
 with (args.output/"P29_CLASS_METRICS.csv").open("w",newline="") as h:w=csv.DictWriter(h,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
 return result
def main():print(json.dumps(run(make_parser().parse_args()),sort_keys=True))
if __name__=="__main__":main()
