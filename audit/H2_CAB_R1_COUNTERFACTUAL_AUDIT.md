# H2 CAB-LoRA R1 Counterfactual Audit

{
  "audit_batches": 16,
  "computational_form": "one counterfactual image per source image, preserving up to four selected token footprints; no per-token full-image explosion",
  "construction": "cyclic peer context from same VisA source batch; union of selected footprints plus 14px halo protected; 14px smooth distance transition",
  "context_changed": "PASS",
  "deterministic": true,
  "local_footprint_identity": "PASS",
  "no_target_information": true,
  "pair_rows": [
    {
      "batch_index": 1,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 0.6642932891845703,
      "protected_pixel_fraction": 0.008765522279035792,
      "token_indices": [
        654,
        655
      ]
    },
    {
      "batch_index": 2,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 0.8866742849349976,
      "protected_pixel_fraction": 0.013148283418553688,
      "token_indices": [
        910,
        1020
      ]
    },
    {
      "batch_index": 4,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 0.6835559010505676,
      "protected_pixel_fraction": 0.017531044558071585,
      "token_indices": [
        507,
        508,
        545,
        616
      ]
    },
    {
      "batch_index": 5,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 1.211949110031128,
      "protected_pixel_fraction": 0.016070124178232285,
      "token_indices": [
        7,
        8,
        81,
        84
      ]
    },
    {
      "batch_index": 0,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 1.294755260149638,
      "protected_pixel_fraction": 0.015339663988312637,
      "token_indices": [
        696,
        770,
        772
      ]
    },
    {
      "batch_index": 1,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 0.7852156162261963,
      "protected_pixel_fraction": 0.017531044558071585,
      "token_indices": [
        829,
        834,
        835,
        866
      ]
    },
    {
      "batch_index": 3,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 0.9290338158607483,
      "protected_pixel_fraction": 0.006574141709276844,
      "token_indices": [
        1272
      ]
    },
    {
      "batch_index": 4,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 1.149045467376709,
      "protected_pixel_fraction": 0.02556610664718773,
      "token_indices": [
        507,
        512,
        577,
        584
      ]
    },
    {
      "batch_index": 5,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 1.6424894332885742,
      "protected_pixel_fraction": 0.015339663988312637,
      "token_indices": [
        581,
        728,
        729
      ]
    },
    {
      "batch_index": 0,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 0.6960146427154541,
      "protected_pixel_fraction": 0.010226442658875092,
      "token_indices": [
        531,
        567
      ]
    },
    {
      "batch_index": 1,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 0.6519634127616882,
      "protected_pixel_fraction": 0.006574141709276844,
      "token_indices": [
        392
      ]
    },
    {
      "batch_index": 4,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 0.8088458180427551,
      "protected_pixel_fraction": 0.006574141709276844,
      "token_indices": [
        232
      ]
    },
    {
      "batch_index": 5,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 0.8747798204421997,
      "protected_pixel_fraction": 0.01095690284879474,
      "token_indices": [
        1043,
        1079,
        1080
      ]
    },
    {
      "batch_index": 4,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 1.9576458930969238,
      "protected_pixel_fraction": 0.02191380569758948,
      "token_indices": [
        683,
        686,
        723,
        874
      ]
    },
    {
      "batch_index": 2,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 0.7593883275985718,
      "protected_pixel_fraction": 0.020452885317750184,
      "token_indices": [
        317,
        388,
        392,
        429
      ]
    },
    {
      "batch_index": 4,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 0.5431910753250122,
      "protected_pixel_fraction": 0.015339663988312637,
      "token_indices": [
        488,
        525,
        527,
        562
      ]
    },
    {
      "batch_index": 0,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 0.817355751991272,
      "protected_pixel_fraction": 0.015339663988312637,
      "token_indices": [
        503,
        614,
        615
      ]
    },
    {
      "batch_index": 1,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 0.936643660068512,
      "protected_pixel_fraction": 0.006574141709276844,
      "token_indices": [
        942
      ]
    },
    {
      "batch_index": 2,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 1.431193470954895,
      "protected_pixel_fraction": 0.008765522279035792,
      "token_indices": [
        621,
        622
      ]
    },
    {
      "batch_index": 3,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 1.0380308628082275,
      "protected_pixel_fraction": 0.018991964937910884,
      "token_indices": [
        26,
        137,
        171,
        173
      ]
    },
    {
      "batch_index": 1,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 0.7909759283065796,
      "protected_pixel_fraction": 0.013148283418553688,
      "token_indices": [
        576,
        578,
        615
      ]
    },
    {
      "batch_index": 4,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 1.0050389766693115,
      "protected_pixel_fraction": 0.010226442658875092,
      "token_indices": [
        388,
        424
      ]
    },
    {
      "batch_index": 5,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 1.0503647327423096,
      "protected_pixel_fraction": 0.02410518626734843,
      "token_indices": [
        869,
        903,
        977,
        983
      ]
    },
    {
      "batch_index": 0,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 0.6548975706100464,
      "protected_pixel_fraction": 0.019722425127830533,
      "token_indices": [
        494,
        531,
        870,
        872
      ]
    },
    {
      "batch_index": 1,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 1.3624238967895508,
      "protected_pixel_fraction": 0.016070124178232285,
      "token_indices": [
        160,
        161,
        196,
        199
      ]
    },
    {
      "batch_index": 2,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 1.192144513130188,
      "protected_pixel_fraction": 0.015339663988312637,
      "token_indices": [
        397,
        434,
        471,
        473
      ]
    },
    {
      "batch_index": 5,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 0.38952650129795074,
      "protected_pixel_fraction": 0.02191380569758948,
      "token_indices": [
        718,
        752,
        862,
        863
      ]
    },
    {
      "batch_index": 4,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 0.7239676713943481,
      "protected_pixel_fraction": 0.006574141709276844,
      "token_indices": [
        493
      ]
    },
    {
      "batch_index": 2,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 0.7358407378196716,
      "protected_pixel_fraction": 0.017531044558071585,
      "token_indices": [
        874,
        875,
        906,
        943
      ]
    },
    {
      "batch_index": 0,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 0.7209478616714478,
      "protected_pixel_fraction": 0.02191380569758948,
      "token_indices": [
        424,
        460,
        535,
        642
      ]
    },
    {
      "batch_index": 2,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 0.9682272672653198,
      "protected_pixel_fraction": 0.015339663988312637,
      "token_indices": [
        635,
        636,
        638,
        639
      ]
    },
    {
      "batch_index": 3,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 0.921516478061676,
      "protected_pixel_fraction": 0.02337472607742878,
      "token_indices": [
        652,
        656,
        727,
        762
      ]
    },
    {
      "batch_index": 0,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 0.8638924360275269,
      "protected_pixel_fraction": 0.02410518626734843,
      "token_indices": [
        212,
        361,
        363,
        427
      ]
    },
    {
      "batch_index": 2,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 0.7159535884857178,
      "protected_pixel_fraction": 0.016800584368151936,
      "token_indices": [
        948,
        949,
        985,
        1061
      ]
    },
    {
      "batch_index": 4,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 1.2453227043151855,
      "protected_pixel_fraction": 0.020452885317750184,
      "token_indices": [
        574,
        575,
        646,
        651
      ]
    },
    {
      "batch_index": 5,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 1.25113183259964,
      "protected_pixel_fraction": 0.015339663988312637,
      "token_indices": [
        509,
        583,
        620,
        621
      ]
    },
    {
      "batch_index": 4,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 1.1201591491699219,
      "protected_pixel_fraction": 0.020452885317750184,
      "token_indices": [
        318,
        391,
        428,
        434
      ]
    },
    {
      "batch_index": 5,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 0.8540551066398621,
      "protected_pixel_fraction": 0.006574141709276844,
      "token_indices": [
        642
      ]
    },
    {
      "batch_index": 0,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 0.6449861526489258,
      "protected_pixel_fraction": 0.026296566837107377,
      "token_indices": [
        468,
        472,
        613,
        623
      ]
    },
    {
      "batch_index": 1,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 1.2560452222824097,
      "protected_pixel_fraction": 0.017531044558071585,
      "token_indices": [
        731,
        732,
        733,
        800
      ]
    },
    {
      "batch_index": 2,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 0.6195543706417084,
      "protected_pixel_fraction": 0.015339663988312637,
      "token_indices": [
        683,
        685,
        722,
        723
      ]
    },
    {
      "batch_index": 4,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 1.3027890920639038,
      "protected_pixel_fraction": 0.008765522279035792,
      "token_indices": [
        562,
        563
      ]
    },
    {
      "batch_index": 3,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 0.6654163599014282,
      "protected_pixel_fraction": 0.019722425127830533,
      "token_indices": [
        341,
        414,
        415,
        433
      ]
    },
    {
      "batch_index": 4,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 0.7045341730117798,
      "protected_pixel_fraction": 0.013148283418553688,
      "token_indices": [
        352,
        389,
        426,
        463
      ]
    },
    {
      "batch_index": 5,
      "donor_rule": "cyclic next image in same VisA source batch",
      "footprint_max_abs_diff": 0.0,
      "outside_footprint_mean_abs_diff": 1.1933887004852295,
      "protected_pixel_fraction": 0.02191380569758948,
      "token_indices": [
        423,
        426,
        571,
        572
      ]
    }
  ],
  "protocol_id": "H2_CAB_LORA_R1_BOUNDED",
  "red_team": {
    "chosen_peer_context": "same-source cyclic peer with protected halo and smooth transition; no labels or category names choose the donor",
    "crop_or_resize": "rejected: can alter the protected local footprint",
    "hard outside-footprint swap": "rejected: creates a direct seam adjacent to the protected token",
    "hard peer-image replacement": "rejected: same seam risk and stronger synthetic cue",
    "naive spatial roll": "rejected: wrap-around border cue"
  },
  "rows": [
    {
      "batch_index": 0,
      "context_change_mean": 0.0,
      "context_change_p95": 0.0,
      "file_names": [
        "pcb1/Data/Images/Normal/0417.JPG",
        "pcb2/Data/Images/Normal/0935.JPG",
        "macaroni1/Data/Images/Normal/0178.JPG",
        "pcb2/Data/Images/Normal/0968.JPG",
        "capsules/Data/Images/Anomaly/040.JPG",
        "capsules/Data/Images/Normal/026.JPG"
      ],
      "footprint_max_abs_diff": 0.0,
      "pair_count": 0,
      "positions": [
        [],
        [],
        [],
        [],
        [],
        []
      ],
      "protected_pixel_fraction_mean": 0.0,
      "selected_token_count": 0,
      "token_count": 0
    },
    {
      "batch_index": 1,
      "context_change_mean": 0.8903295993804932,
      "context_change_p95": 1.211949110031128,
      "file_names": [
        "pipe_fryum/Data/Images/Normal/050.JPG",
        "pcb1/Data/Images/Anomaly/061.JPG",
        "pcb4/Data/Images/Anomaly/077.JPG",
        "chewinggum/Data/Images/Normal/352.JPG",
        "pcb3/Data/Images/Anomaly/003.JPG",
        "capsules/Data/Images/Anomaly/064.JPG"
      ],
      "footprint_max_abs_diff": 0.0,
      "pair_count": 4,
      "positions": [
        [],
        [
          654,
          655
        ],
        [
          910,
          1020
        ],
        [],
        [
          507,
          508,
          545,
          616
        ],
        [
          7,
          8,
          81,
          84
        ]
      ],
      "protected_pixel_fraction_mean": 0.013878743608473337,
      "selected_token_count": 12,
      "token_count": 12
    },
    {
      "batch_index": 2,
      "context_change_mean": 1.1651874820391337,
      "context_change_p95": 1.6424894332885742,
      "file_names": [
        "candle/Data/Images/Anomaly/025.JPG",
        "pcb1/Data/Images/Anomaly/020.JPG",
        "macaroni2/Data/Images/Normal/0294.JPG",
        "candle/Data/Images/Anomaly/039.JPG",
        "candle/Data/Images/Anomaly/088.JPG",
        "pipe_fryum/Data/Images/Anomaly/025.JPG"
      ],
      "footprint_max_abs_diff": 0.0,
      "pair_count": 5,
      "positions": [
        [
          696,
          770,
          772
        ],
        [
          829,
          834,
          835,
          866
        ],
        [],
        [
          1272
        ],
        [
          507,
          512,
          577,
          584
        ],
        [
          581,
          728,
          729
        ]
      ],
      "protected_pixel_fraction_mean": 0.016070124178232288,
      "selected_token_count": 15,
      "token_count": 15
    },
    {
      "batch_index": 3,
      "context_change_mean": 0.7824539967945644,
      "context_change_p95": 0.8747798204421997,
      "file_names": [
        "cashew/Data/Images/Anomaly/071.JPG",
        "macaroni1/Data/Images/Anomaly/064.JPG",
        "pcb3/Data/Images/Normal/0484.JPG",
        "pcb3/Data/Images/Normal/0871.JPG",
        "macaroni1/Data/Images/Anomaly/015.JPG",
        "candle/Data/Images/Anomaly/037.JPG"
      ],
      "footprint_max_abs_diff": 0.0,
      "pair_count": 4,
      "positions": [
        [
          531,
          567
        ],
        [
          392
        ],
        [],
        [],
        [
          232
        ],
        [
          1043,
          1079,
          1080
        ]
      ],
      "protected_pixel_fraction_mean": 0.00858290723155588,
      "selected_token_count": 7,
      "token_count": 7
    },
    {
      "batch_index": 4,
      "context_change_mean": 1.9576458930969238,
      "context_change_p95": 1.9576458930969238,
      "file_names": [
        "capsules/Data/Images/Anomaly/006.JPG",
        "capsules/Data/Images/Normal/324.JPG",
        "pcb3/Data/Images/Normal/0024.JPG",
        "pcb3/Data/Images/Normal/0159.JPG",
        "pcb1/Data/Images/Anomaly/064.JPG",
        "pcb4/Data/Images/Normal/0619.JPG"
      ],
      "footprint_max_abs_diff": 0.0,
      "pair_count": 1,
      "positions": [
        [],
        [],
        [],
        [],
        [
          683,
          686,
          723,
          874
        ],
        []
      ],
      "protected_pixel_fraction_mean": 0.02191380569758948,
      "selected_token_count": 4,
      "token_count": 4
    },
    {
      "batch_index": 5,
      "context_change_mean": 0.651289701461792,
      "context_change_p95": 0.7593883275985718,
      "file_names": [
        "pcb2/Data/Images/Normal/0427.JPG",
        "macaroni1/Data/Images/Normal/0631.JPG",
        "pcb4/Data/Images/Anomaly/028.JPG",
        "capsules/Data/Images/Anomaly/038.JPG",
        "pcb4/Data/Images/Anomaly/033.JPG",
        "pipe_fryum/Data/Images/Anomaly/088.JPG"
      ],
      "footprint_max_abs_diff": 0.0,
      "pair_count": 2,
      "positions": [
        [],
        [],
        [
          317,
          388,
          392,
          429
        ],
        [],
        [
          488,
          525,
          527,
          562
        ],
        []
      ],
      "protected_pixel_fraction_mean": 0.01789627465303141,
      "selected_token_count": 8,
      "token_count": 8
    },
    {
      "batch_index": 6,
      "context_change_mean": 1.040322130918503,
      "context_change_p95": 1.431193470954895,
      "file_names": [
        "capsules/Data/Images/Anomaly/028.JPG",
        "pcb1/Data/Images/Anomaly/023.JPG",
        "macaroni2/Data/Images/Anomaly/085.JPG",
        "capsules/Data/Images/Anomaly/062.JPG",
        "fryum/Data/Images/Normal/307.JPG",
        "pipe_fryum/Data/Images/Normal/148.JPG"
      ],
      "footprint_max_abs_diff": 0.0,
      "pair_count": 4,
      "positions": [
        [
          503,
          614,
          615
        ],
        [
          942
        ],
        [
          621,
          622
        ],
        [
          26,
          137,
          171,
          173
        ],
        [],
        []
      ],
      "protected_pixel_fraction_mean": 0.01241782322863404,
      "selected_token_count": 10,
      "token_count": 10
    },
    {
      "batch_index": 7,
      "context_change_mean": 0.9538294076919556,
      "context_change_p95": 1.0503647327423096,
      "file_names": [
        "cashew/Data/Images/Normal/272.JPG",
        "pcb3/Data/Images/Anomaly/090.JPG",
        "pcb1/Data/Images/Normal/0208.JPG",
        "macaroni2/Data/Images/Anomaly/019.JPG",
        "candle/Data/Images/Anomaly/070.JPG",
        "capsules/Data/Images/Anomaly/080.JPG"
      ],
      "footprint_max_abs_diff": 0.0,
      "pair_count": 3,
      "positions": [
        [],
        [
          576,
          578,
          615
        ],
        [],
        [],
        [
          388,
          424
        ],
        [
          869,
          903,
          977,
          983
        ]
      ],
      "protected_pixel_fraction_mean": 0.01582663744825907,
      "selected_token_count": 9,
      "token_count": 9
    },
    {
      "batch_index": 8,
      "context_change_mean": 0.899748120456934,
      "context_change_p95": 1.3624238967895508,
      "file_names": [
        "pcb4/Data/Images/Anomaly/063.JPG",
        "chewinggum/Data/Images/Anomaly/085.JPG",
        "capsules/Data/Images/Anomaly/014.JPG",
        "capsules/Data/Images/Normal/580.JPG",
        "macaroni2/Data/Images/Normal/0631.JPG",
        "pipe_fryum/Data/Images/Anomaly/074.JPG"
      ],
      "footprint_max_abs_diff": 0.0,
      "pair_count": 4,
      "positions": [
        [
          494,
          531,
          870,
          872
        ],
        [
          160,
          161,
          196,
          199
        ],
        [
          397,
          434,
          471,
          473
        ],
        [],
        [],
        [
          718,
          752,
          862,
          863
        ]
      ],
      "protected_pixel_fraction_mean": 0.018261504747991236,
      "selected_token_count": 16,
      "token_count": 16
    },
    {
      "batch_index": 9,
      "context_change_mean": 0.7239676713943481,
      "context_change_p95": 0.7239676713943481,
      "file_names": [
        "macaroni1/Data/Images/Normal/0685.JPG",
        "fryum/Data/Images/Normal/377.JPG",
        "pcb3/Data/Images/Normal/0294.JPG",
        "pcb4/Data/Images/Normal/0674.JPG",
        "macaroni1/Data/Images/Anomaly/063.JPG",
        "pcb3/Data/Images/Normal/0694.JPG"
      ],
      "footprint_max_abs_diff": 0.0,
      "pair_count": 1,
      "positions": [
        [],
        [],
        [],
        [],
        [
          493
        ],
        []
      ],
      "protected_pixel_fraction_mean": 0.006574141709276844,
      "selected_token_count": 1,
      "token_count": 1
    },
    {
      "batch_index": 10,
      "context_change_mean": 0.7358407378196716,
      "context_change_p95": 0.7358407378196716,
      "file_names": [
        "pcb1/Data/Images/Normal/0215.JPG",
        "pcb3/Data/Images/Normal/1002.JPG",
        "pcb2/Data/Images/Anomaly/059.JPG",
        "pcb1/Data/Images/Normal/0784.JPG",
        "pcb3/Data/Images/Normal/0569.JPG",
        "pcb3/Data/Images/Normal/0735.JPG"
      ],
      "footprint_max_abs_diff": 0.0,
      "pair_count": 1,
      "positions": [
        [],
        [],
        [
          874,
          875,
          906,
          943
        ],
        [],
        [],
        []
      ],
      "protected_pixel_fraction_mean": 0.017531044558071585,
      "selected_token_count": 4,
      "token_count": 4
    },
    {
      "batch_index": 11,
      "context_change_mean": 0.8702305356661478,
      "context_change_p95": 0.9682272672653198,
      "file_names": [
        "pcb1/Data/Images/Anomaly/086.JPG",
        "pcb1/Data/Images/Normal/0619.JPG",
        "pcb3/Data/Images/Anomaly/044.JPG",
        "chewinggum/Data/Images/Anomaly/073.JPG",
        "pipe_fryum/Data/Images/Normal/093.JPG",
        "chewinggum/Data/Images/Normal/050.JPG"
      ],
      "footprint_max_abs_diff": 0.0,
      "pair_count": 3,
      "positions": [
        [
          424,
          460,
          535,
          642
        ],
        [],
        [
          635,
          636,
          638,
          639
        ],
        [
          652,
          656,
          727,
          762
        ],
        [],
        []
      ],
      "protected_pixel_fraction_mean": 0.020209398587776966,
      "selected_token_count": 12,
      "token_count": 12
    },
    {
      "batch_index": 12,
      "context_change_mean": 1.0190751403570175,
      "context_change_p95": 1.2511318922042847,
      "file_names": [
        "chewinggum/Data/Images/Anomaly/063.JPG",
        "cashew/Data/Images/Normal/050.JPG",
        "macaroni1/Data/Images/Anomaly/044.JPG",
        "pcb4/Data/Images/Normal/0964.JPG",
        "pipe_fryum/Data/Images/Anomaly/040.JPG",
        "candle/Data/Images/Anomaly/092.JPG"
      ],
      "footprint_max_abs_diff": 0.0,
      "pair_count": 4,
      "positions": [
        [
          212,
          361,
          363,
          427
        ],
        [],
        [
          948,
          949,
          985,
          1061
        ],
        [],
        [
          574,
          575,
          646,
          651
        ],
        [
          509,
          583,
          620,
          621
        ]
      ],
      "protected_pixel_fraction_mean": 0.019174579985390797,
      "selected_token_count": 16,
      "token_count": 16
    },
    {
      "batch_index": 13,
      "context_change_mean": 1.06693834066391,
      "context_change_p95": 1.1201591491699219,
      "file_names": [
        "cashew/Data/Images/Normal/352.JPG",
        "fryum/Data/Images/Normal/006.JPG",
        "candle/Data/Images/Normal/0457.JPG",
        "pcb1/Data/Images/Normal/0755.JPG",
        "capsules/Data/Images/Anomaly/074.JPG",
        "pcb3/Data/Images/Anomaly/096.JPG"
      ],
      "footprint_max_abs_diff": 0.0,
      "pair_count": 2,
      "positions": [
        [],
        [],
        [],
        [],
        [
          318,
          391,
          428,
          434
        ],
        [
          642
        ]
      ],
      "protected_pixel_fraction_mean": 0.013513513513513514,
      "selected_token_count": 5,
      "token_count": 5
    },
    {
      "batch_index": 14,
      "context_change_mean": 0.9062800833157131,
      "context_change_p95": 1.3027890920639038,
      "file_names": [
        "pcb1/Data/Images/Anomaly/046.JPG",
        "capsules/Data/Images/Anomaly/056.JPG",
        "cashew/Data/Images/Anomaly/087.JPG",
        "pcb4/Data/Images/Normal/0804.JPG",
        "pcb3/Data/Images/Anomaly/054.JPG",
        "candle/Data/Images/Normal/0024.JPG"
      ],
      "footprint_max_abs_diff": 0.0,
      "pair_count": 4,
      "positions": [
        [
          468,
          472,
          613,
          623
        ],
        [
          731,
          732,
          733,
          800
        ],
        [
          683,
          685,
          722,
          723
        ],
        [],
        [
          562,
          563
        ],
        []
      ],
      "protected_pixel_fraction_mean": 0.016983199415631846,
      "selected_token_count": 14,
      "token_count": 14
    },
    {
      "batch_index": 15,
      "context_change_mean": 0.8544464111328125,
      "context_change_p95": 1.1933887004852295,
      "file_names": [
        "macaroni2/Data/Images/Normal/0900.JPG",
        "macaroni2/Data/Images/Normal/0784.JPG",
        "macaroni2/Data/Images/Anomaly/060.JPG",
        "chewinggum/Data/Images/Anomaly/033.JPG",
        "pcb4/Data/Images/Anomaly/057.JPG",
        "chewinggum/Data/Images/Anomaly/054.JPG"
      ],
      "footprint_max_abs_diff": 0.0,
      "pair_count": 3,
      "positions": [
        [],
        [],
        [],
        [
          341,
          414,
          415,
          433
        ],
        [
          352,
          389,
          426,
          463
        ],
        [
          423,
          426,
          571,
          572
        ]
      ],
      "protected_pixel_fraction_mean": 0.018261504747991233,
      "selected_token_count": 12,
      "token_count": 12
    }
  ],
  "selection": "existing source training masks only; zero-footprint native token intersecting existing 7x7 near-background",
  "source_domain_compatible": true,
  "summary": {
    "context_change_mean": 0.9678390168126613,
    "context_change_p95": 1.40292500535647,
    "footprint_max_abs_diff": 0.0,
    "selected_token_count": 145
  }
}
