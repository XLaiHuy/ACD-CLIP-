import logging
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_data_root() -> Path:
    """Resolve one repository-facing dataset root for every dataset constant."""
    override = os.environ.get("ACDCLIP_DATA_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    repository_data = REPO_ROOT / "data"
    if repository_data.exists():
        return repository_data.resolve()
    legacy_data = REPO_ROOT.parent / "data"
    if legacy_data.exists():
        return legacy_data.resolve()
    return repository_data.resolve()


DATA_ROOT = resolve_data_root()
BASE_PATH = str(REPO_ROOT).replace("\\", "/")


def log_data_root(logger: logging.Logger | None = None) -> str:
    """Log and return the resolved data root for startup/preflight reports."""
    resolved = str(DATA_ROOT)
    (logger or logging.getLogger(__name__)).info("ACDCLIP_DATA_ROOT=%s", resolved)
    return resolved


def _data_path(*parts: str) -> str:
    return str(DATA_ROOT.joinpath(*parts))

DATA_PATH = {
    "Brain": _data_path("MedAD", "Brain_AD", "test"),
    "Liver": _data_path("MedAD", "Liver_AD", "test"),
    "Retina": _data_path("MedAD", "Retina_RESC_AD", "test"),
    "Colon_clinicDB": _data_path("Colon", "CVC-ClinicDB"),
    "Colon_colonDB": _data_path("Colon", "CVC-ColonDB"),
    "Colon_cvc300": _data_path("Colon", "CVC-300"),
    "Colon_Kvasir": _data_path("Colon", "Kvasir"),
    "BTAD": _data_path("BTech_Dataset_transformed"),
    "MPDD": _data_path("MPDD"),
    "MVTec": _data_path("mvtec_ad"),
    "VisA": _data_path("VisA_20220922"),
    "RSDD": _data_path("RSDD")
}

# Medical validation is never used for training.  Brain/Liver/Retina have
# provider validation directories; the three Colon datasets receive a
# deterministic manifest split prepared per run by
# tools/prepare_phase4_medical_splits.py.
MEDICAL_EVAL_PATHS = {
    "Brain": {
        "val": _data_path("MedAD", "Brain_AD", "valid"),
        "test": DATA_PATH["Brain"],
    },
    "Liver": {
        "val": _data_path("MedAD", "Liver_AD", "valid"),
        "test": DATA_PATH["Liver"],
    },
    "Retina": {
        "val": _data_path("MedAD", "Retina_RESC_AD", "val"),
        "test": DATA_PATH["Retina"],
    },
    "Colon_clinicDB": {"val": DATA_PATH["Colon_clinicDB"], "test": DATA_PATH["Colon_clinicDB"]},
    "Colon_colonDB": {"val": DATA_PATH["Colon_colonDB"], "test": DATA_PATH["Colon_colonDB"]},
    "Colon_Kvasir": {"val": DATA_PATH["Colon_Kvasir"], "test": DATA_PATH["Colon_Kvasir"]},
}

CLASS_NAMES = {
    "Brain": ["Brain"],
    "Liver": ["Liver"],
    "Retina": ["Retina"],
    "Colon_clinicDB": ["Colon_clinicDB"],
    "Colon_colonDB": ["Colon_colonDB"],
    "Colon_Kvasir": ["Colon_Kvasir"],
    "Colon_cvc300": ["CVC-300"],
    "MVTec": [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "transistor",
        "toothbrush",
        "wood",
        "zipper",
    ],
    "VisA": [
        "candle",
        "pcb3",
        "capsules",
        "pipe_fryum",
        "pcb4",
        "macaroni2",
        "pcb2",
        "chewinggum",
        "macaroni1",
        "cashew",
        "fryum",
        "pcb1",
    ],
    "MPDD": [
        "connector",
        "tubes",
        "metal_plate",
        "bracket_white",
        "bracket_brown",
        "bracket_black",
    ],
    "BTAD": ["01", "02", "03"],
    "RSDD": [
        "Dent",
        "Crush",
        "Scratch",
        "Damage"
    ]
}
DOMAINS = {
    "VisA": "Industrial",
    "BTAD": "Industrial",
    "MPDD": "Industrial",
    "MVTec": "Industrial",
    "RSDD": "Industrial",
    "Brain": "Medical",
    "Liver": "Medical",
    "Retina": "Medical",
    "Colon_clinicDB": "Medical",
    "Colon_colonDB": "Medical",
    "Colon_Kvasir": "Medical",
    "Colon_cvc300": "Medical",
}
REAL_NAMES = {
    "Brain": {"Brain": "scan"},
    "Liver": {"Liver": "scan"},
    "Retina": {"Retina": "scan"},
    "MVTec": {
        "bottle": "dark bottle",
        "cable": "top view of three cables",
        "capsule": "black and orange capsule",
        "carpet": "gray carpet",
        "grid": "metal or plastic mesh",
        "hazelnut": "single brown hazelnut",
        "leather": "brown leather",
        "metal_nut": "metal nut which has four notched edges",
        "pill": "oval white pill with small red speckles and the letters 'FF' engraved",
        "screw": "screw",
        "tile": "speckled tile surface",
        "transistor": "a three-legged transistor placed vertically",
        "toothbrush": "toothbrush head",
        "wood": "wood surface",
        "zipper": "a black zipper",
    },
    "VisA": {
        "candle": "candle",
        "pcb3": "infrared sensor pcb module",
        "capsules": "capsules",
        "pipe_fryum": "pipe-shaped fryum",
        "pcb4": "battery charging pcb module",
        "macaroni2": "scattered yellow macaroni",
        "pcb2": "integrated circuits board",
        "chewinggum": "chewing gum",
        "macaroni1": "orange macaroni",
        "cashew": "cashew nut",
        "fryum": "wheel-shaped fryum snack",
        "pcb1": "dual ultrasonic distance sensor pcb module",
    },
    "Colon_clinicDB": {
        "Colon_clinicDB": "colon endoscopy image",
    },
    "Colon_colonDB": {
        "Colon_colonDB": "colon endoscopy image",
    },
    "Colon_cvc300": {"CVC-300": "colon endoscopy image"},
    "Colon_Kvasir": {"Colon_Kvasir": "colon endoscopy image"},
    "MPDD": {
        "connector": "metal clamps with black adjustment knobs",
        "tubes": "scattered metal objects",
        "metal_plate": "blue rectangular metal plate with a notch on one side",
        "bracket_white": "white, elongated triangular metal bracket with a smooth, matte finish",
        "bracket_brown": "brown L-shaped metal bracket with smooth, glossy finish and multiple mounting holes along its arms",
        "bracket_black": "black ornamental metal bracket with spiral design attached to a rectangular frame",
    },
    "BTAD": {
        "01": "Bright concentric rings in neon yellow and blue tones against a dark blue background, resembling a stylized wave or energy field radiating outward.",
        "02": "vertical fabric lines in warm, dusty pink and beige tones",
        "03": "oval concentric circular rings in gradient shades of blue and white",
    },
    "RSDD": {
        "Dent": "Dent",
        "Crush": "Crush",
        "Scratch": "Scratch",
        "Damage": "Damage"
    }
}
PROMPTS = {
    "prompt_normal": ["{}", "a {}", "the {}"],
    "prompt_abnormal": [
        "a damaged {}",
        "a broken {}",
        "a {} with flaw",
        "a {} with defect",
        "a {} with damage",
    ],
    "prompt_templates": [
        "{}.",
        "a photo of {}.",
    ],
}
