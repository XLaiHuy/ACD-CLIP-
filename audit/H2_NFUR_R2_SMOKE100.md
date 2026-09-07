# H2 NFUR-R2 source smoke (100 attempted steps)

{
  "attempted_steps": 100,
  "catastrophic_screen": "PASS",
  "control": {
    "image_ap": 86.86788082122803,
    "image_auroc": 80.0000011920929,
    "pixel_ap": 59.28671360015869,
    "pixel_auroc": 99.42891597747803,
    "pixels": 6439776,
    "recall_at_1pct_normal_fpr": 88.55169373208905,
    "samples": 24
  },
  "nfur": {
    "image_ap": 79.64995503425598,
    "image_auroc": 61.481475830078125,
    "pixel_ap": 65.80367088317871,
    "pixel_auroc": 99.25740957260132,
    "pixels": 6439776,
    "recall_at_1pct_normal_fpr": 85.06574597810916,
    "samples": 24
  },
  "nfur_diagnostics": {
    "activity": true,
    "delta_native_abs_mean": 0.24527046084403992,
    "delta_native_abs_p95": 0.24752137064933777,
    "delta_requires_grad": false,
    "enabled": true,
    "gated_correction_abs_mean": 0.06544507294893265,
    "gated_correction_abs_p95": 0.10436227917671204,
    "input_channels": 772,
    "margin_uncertainty_mean": 0.05218800529837608,
    "native_grid": [
      37,
      37
    ],
    "stage_disagreement_mean": 0.48586779832839966,
    "uncertainty_mean": 0.2690278887748718,
    "uncertainty_p95": 0.5187852382659912
  },
  "nfur_final_conv_abs_sum": 0.617178738117218,
  "nfur_nonzero_activity": true,
  "numerical_validity": true,
  "paired_batches": true,
  "protocol_id": "H2_NFUR_R2_E20_MEDICAL_SELECTED",
  "user_override": "smoke failure does not block full training; identity-off failure remains blocking"
}
