import logging
import torch
import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import sys


CURRENT_DIR = Path(__file__).resolve().parent
DISK_REPO_PATH = CURRENT_DIR / "disk"
if str(DISK_REPO_PATH) not in sys.path:
    sys.path.append(str(DISK_REPO_PATH))

try:
    from disk import DISK
except ImportError:
    from disk.model import DISK

logger = logging.getLogger(__name__)

class PresetRecognizer:
    """
    Classe para reconhecimento de presets de câmera usando DISK.
    Pipeline: Pré-processamento -> Extração DISK -> Matching L2
    """
    def __init__(
        self,
        model_path: str = "algorithms/disk/depth-save.pth",
        good_match_ratio: float = 0.8, 
        min_good_matches: int = 15,
        target_size: Tuple[int, int] = (640, 480),
        top_k: int = 2048,
    ) -> None:
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Usando dispositivo: {self.device}")
        self.good_match_ratio = good_match_ratio
        self.min_good_matches = min_good_matches
        self.target_size = target_size
        self.top_k = top_k

        self.model = DISK().to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        if 'extractor' in state_dict:
            self.model.load_state_dict(state_dict['extractor'])
        else:
            self.model.load_state_dict(state_dict)
        self.model.eval()

        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        self.descritores_presets: Dict[str, np.ndarray] = {}

    def _preprocess_image(self, imagem: np.ndarray) -> torch.Tensor:
        if imagem is None:
            raise ValueError("Imagem inválida")
        
        img_res = cv2.resize(imagem, self.target_size)
        if len(img_res.shape) == 3:
            img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
        
        img_tensor = torch.from_numpy(img_res).float().permute(2, 0, 1) / 255.0
        return img_tensor.unsqueeze(0).to(self.device)

    def _extract_features(self, imagem: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extrai keypoints e descritores usando DISK."""
        img_tensor = self._preprocess_image(imagem)
        
        with torch.no_grad():
            batched_features = self.model.features(
                img_tensor, 
                kind='nms', 
                n=self.top_k,
                window_size=5
            )

            features = batched_features[0] 
            kps = features.kp.cpu().numpy()
            descs = features.desc.cpu().numpy()
            
        return kps, descs

    def configurar_presets(self, presets: Dict[str, np.ndarray]) -> None:
        self.descritores_presets = {}
        print(f"Configurando {len(presets)} presets com DISK...")
        for i, (nome, img) in enumerate(presets.items()):
            try:
                print(f"[{i+1}/{len(presets)}] Extraindo features do preset: {nome}")
                _, descs = self._extract_features(img)
                if descs is not None:
                    self.descritores_presets[nome] = descs
            except Exception as e:
                print(f"Erro no preset {nome}: {e}")

    def identificar_preset(self, imagem: np.ndarray) -> Tuple[Optional[str], float]:
        _, descs_nova = self._extract_features(imagem)
        
        if descs_nova is None:
            return None, 0.0

        melhor_preset = None
        melhor_score = 0.0

        for nome, descs_ref in self.descritores_presets.items():
            # Matching K-Nearest Neighbors
            matches = self.bf.knnMatch(descs_nova, descs_ref, k=2)
            
            # Lowe's Ratio Test
            good_matches = [m for m, n in matches if m.distance < self.good_match_ratio * n.distance]
            score = float(len(good_matches))

            if score > melhor_score:
                melhor_score = score
                melhor_preset = nome

        if melhor_score >= self.min_good_matches:
            print(f"Preset identificado: {melhor_preset} com score {melhor_score}")
            return melhor_preset, melhor_score
        print("Nenhum preset identificado com confiança suficiente.")
        return None, melhor_score