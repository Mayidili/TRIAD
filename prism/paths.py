"""Default data and output locations. Each can be overridden with an environment variable."""
import os

DATA_ROOT = os.environ.get("TRIAD_DATA", "data")
RESULTS_ROOT = os.environ.get("TRIAD_RESULTS", "results")
INTERIM_DIR = os.environ.get("TRIAD_INTERIM", os.path.join(RESULTS_ROOT, "interim"))
RUN_ROOT = os.environ.get("TRIAD_RUN", os.path.join(RESULTS_ROOT, "triad"))
RUN_DATA = os.path.join(RUN_ROOT, "data")
RAD_POOL = os.environ.get("TRIAD_RAD_POOL", os.path.join(DATA_ROOT, "duke_feature_pool.npz"))
KIN_NPZ = os.environ.get("TRIAD_KIN", os.path.join(RUN_ROOT, "data/duke_kinetic.npz"))
ABLATION_DIR = os.environ.get("TRIAD_ABLATIONS", os.path.join(RUN_ROOT, "ablations"))
SEG_QA_DIR = os.environ.get("TRIAD_SEG_QA", os.path.join(RUN_ROOT, "segmentation_qa"))
MAMAMIA_DIR = os.environ.get("TRIAD_MAMAMIA", os.path.join(DATA_ROOT, "MAMA-MIA"))
FIGURES_DIR = os.environ.get("TRIAD_FIGURES", "figures")
FONT_DIR = os.environ.get("TRIAD_FONT_DIR", "fonts/times-new-roman")
MEDSAM2_DIR = os.environ.get("MEDSAM2_DIR", "MedSAM2")
MEDSAM2_CKPT = os.environ.get("MEDSAM2_CKPT", os.path.join(MEDSAM2_DIR, "checkpoints/MedSAM2_MRI_LiverLesion.pt"))
MEDSAM2_CFG = os.environ.get("MEDSAM2_CFG", "configs/sam2.1_hiera_t512.yaml")
