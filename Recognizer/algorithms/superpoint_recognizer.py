from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import logging

from algorithms.superpoint.superpoint_pytorch import SuperPoint  # model

logger = logging.getLogger(__name__)

class SuperPointPresetRecognizer:
    def __init__(
        self,
        weights_path: str = "algorithms/superpoint/weights/superpoint_v6_from_tf.pth",
        good_match_ratio: float = 0.75,
        min_good_matches: int = 10,
        target_size: Tuple[int, int] = (640, 480),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        if not (0 < good_match_ratio < 1):
            raise ValueError("good_match_ratio must be in (0, 1).")
        self.good_match_ratio = good_match_ratio
        self.min_good_matches = min_good_matches
        self.target_size = target_size
        self.device = torch.device(device)

        # SuperPoint model
        self.model = SuperPoint().to(self.device).eval()
        state = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(
            state["model_state_dict"] if "model_state_dict" in state else state
        )

        # BF matcher for float descriptors
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        self.descritores_presets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.presets_referencia: Dict[str, np.ndarray] = {}

    def _preprocess_image(self, imagem: np.ndarray) -> np.ndarray:
        if imagem is None:
            raise ValueError("Invalid image: None")
        if len(imagem.shape) == 3:
            imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        return cv2.resize(imagem, self.target_size)

    def _compute_descriptors(
        self, imagem: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if imagem is None:
            return np.empty((0, 2)), None
        img = torch.from_numpy(imagem).float() / 255.0
        if img.ndim == 2:
            img = img.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        else:
            img = img.permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)
        img = img.to(self.device)

        with torch.no_grad():
            pred = self.model({"image": img})

        kpts = pred["keypoints"][0].cpu().numpy()  # shape (N, 2)
        desc = pred["descriptors"][0].cpu().numpy()  # Check actual shape

        # SuperPoint descriptors are (D, H, W) or (N, D) - adjust transpose if needed
        if desc.ndim == 3:  # (D, H, W) format
            desc = desc.reshape(desc.shape[0], -1).T  # -> (H*W, D)
        elif desc.ndim == 2 and desc.shape[0] != kpts.shape[0]:
            desc = desc.T  # Transpose if dimensions swapped

        if desc is None or kpts.shape[0] == 0:
            return np.empty((0, 2)), None

        return kpts, desc

    def configurar_presets(self, presets: Dict[str, np.ndarray]) -> None:
        self.presets_referencia = {}
        for nome, img in presets.items():
            try:
                self.presets_referencia[nome] = self._preprocess_image(img)
            except ValueError:
                continue
        self._computar_descritores_presets()

    def _computar_descritores_presets(self) -> None:
        self.descritores_presets = {}
        for nome_preset, imagem in self.presets_referencia.items():
            kpts, desc = self._compute_descriptors(imagem)
            if desc is None or kpts.shape[0] == 0:
                continue
            self.descritores_presets[nome_preset] = (kpts, desc)

    def _calcular_good_matches(
        self, descriptors_novos: np.ndarray, descriptors_preset: np.ndarray
    ) -> int:
        if descriptors_novos is None or descriptors_preset is None:
            return 0

        # Validate shapes match
        if descriptors_novos.shape[1] != descriptors_preset.shape[1]:
            print(
                f"Descriptor dimension mismatch: {descriptors_novos.shape[1]} vs {descriptors_preset.shape[1]}"
            )
            return 0

        # Ensure both are float32
        desc_new = descriptors_novos.astype(np.float32)
        desc_preset = descriptors_preset.astype(np.float32)

        try:
            knn_matches = self.bf.knnMatch(desc_new, desc_preset, k=2)
        except cv2.error as e:
            print(f"knnMatch error: {e}")
            return 0

        good = []
        for m_n in knn_matches:
            if len(m_n) < 2:
                continue
            m, n = m_n
            if m.distance < self.good_match_ratio * n.distance:
                good.append(m)
        return len(good)

    def identificar_preset(self, imagem: np.ndarray) -> Tuple[Optional[str], float]:
        if not self.descritores_presets:
            return None, 0.0
        img_proc = self._preprocess_image(imagem)
        kpts_new, desc_new = self._compute_descriptors(img_proc)
        if desc_new is None or kpts_new.shape[0] == 0:
            return None, 0.0

        scores = {
            nome: self._calcular_good_matches(desc_new, desc_preset)
            for nome, (_, desc_preset) in self.descritores_presets.items()
        }
        if not scores:
            return None, 0.0
        best_name = max(scores, key=scores.get)
        best_score = float(scores[best_name])
        if best_score >= self.min_good_matches:
            print(f"Preset identificado: {best_name} com score {best_score}")
            return best_name, best_score
        else:
            return None, 0.0