import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from algorithms.r2d2.extract import load_network, extract_multiscale, NonMaxSuppression
from algorithms.r2d2.tools.dataloader import norm_RGB

logger = logging.getLogger(__name__)


class PresetRecognizer:
    """
    Classe para reconhecimento de presets de câmera usando R2D2.
    Pipeline: pré-processamento -> R2D2 extração -> matching com descritores
    """

    def __init__(
        self,
        model_path: str = "algorithms/r2d2/r2d2_WASF_N16.pt",
        good_match_ratio: float = 0.75,
        min_good_matches: int = 10,
        target_size: Tuple[int, int] = (640, 480),
        top_k: int = 5000,
        reliability_thr: float = 0.7,
        repeatability_thr: float = 0.7,
    ) -> None:
        """
        Inicializa o reconhecedor de presets com R2D2.

        Args:
            model_path: caminho para o modelo R2D2.
            good_match_ratio: razão de Lowe no matching (0 < ratio < 1).
            min_good_matches: número mínimo de good matches para identificar um preset.
            target_size: tamanho para redimensionar imagens (largura, altura).
            top_k: número máximo de keypoints a extrair.
            reliability_thr: threshold de confiabilidade para NMS.
            repeatability_thr: threshold de repetibilidade para NMS.
        """
        if not (0 < good_match_ratio < 1):
            raise ValueError("good_match_ratio deve estar em (0, 1).")

        self.good_match_ratio = good_match_ratio
        self.min_good_matches = min_good_matches
        self.target_size = target_size
        self.top_k = top_k

        # Carrega o modelo R2D2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_network(model_path)
        if torch.cuda.is_available():
            self.net = self.net.cuda()

        # Detector NMS
        self.detector = NonMaxSuppression(
            rel_thr=reliability_thr,
            rep_thr=repeatability_thr
        )

        # Matcher - R2D2 usa descritores L2
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        # Descritores dos presets
        self.descritores_presets: Dict[str, np.ndarray] = {}
        self.keypoints_presets: Dict[str, np.ndarray] = {}
        self.presets_referencia: Dict[str, np.ndarray] = {}

    def _preprocess_image(self, imagem: np.ndarray) -> np.ndarray:
        """
        Converte para RGB (se necessário) e redimensiona para target_size.
        """
        if imagem is None:
            raise ValueError("Imagem inválida: None")

        # Converte BGR para RGB se necessário
        if len(imagem.shape) == 3 and imagem.shape[2] == 3:
            imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

        try:
            imagem = cv2.resize(imagem, self.target_size)
        except cv2.error as e:
            raise ValueError(
                f"Falha ao redimensionar imagem de tamanho {imagem.shape}: {e}"
            )

        return imagem

    def _extract_r2d2_features(
        self, imagem: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extrai keypoints e descritores R2D2 de uma imagem.

        Returns:
            (keypoints, descriptors) onde keypoints é Nx2 (x, y) e descriptors é NxD
        """
        if imagem is None or imagem.size == 0:
            return np.array([]), None

        try:
            # Converte para tensor PyTorch
            img_pil = Image.fromarray(imagem)
            img_tensor = norm_RGB(img_pil)[None]  # [1, 3, H, W]

            if torch.cuda.is_available():
                img_tensor = img_tensor.cuda()

            # Extrai features em múltiplas escalas
            with torch.no_grad():
                xys, desc, scores = extract_multiscale(
                    self.net,
                    img_tensor,
                    self.detector,
                    scale_f=2**0.25,
                    min_scale=0.0,
                    max_scale=1,
                    min_size=256,
                    max_size=1024,
                    verbose=False
                )

            xys = xys.cpu().numpy()
            desc = desc.cpu().numpy()
            scores = scores.cpu().numpy()

            # Seleciona top-k keypoints
            if len(scores) > 0:
                idxs = scores.argsort()[-self.top_k or None:]
                keypoints = xys[idxs, :2]  # Pega apenas x, y
                descriptors = desc[idxs]
            else:
                return np.array([]), None

            return keypoints, descriptors

        except Exception as e:
            logger.warning(f"Falha ao extrair features R2D2: {e}")
            return np.array([]), None

    def configurar_presets(self, presets: Dict[str, np.ndarray]) -> None:
        """
        Configura os presets de referência.
        """
        self.presets_referencia = {}
        self.keypoints_presets = {}
        self.descritores_presets = {}

        for nome, img in presets.items():
            if img is None:
                logger.warning(f"Imagem do preset '{nome}' é None; ignorando")
                continue

            try:
                img_proc = self._preprocess_image(img)
                keypoints, descriptors = self._extract_r2d2_features(img_proc)

                if descriptors is None or len(keypoints) == 0:
                    logger.warning(
                        f"Preset '{nome}' não possui descritores suficientes; ignorando"
                    )
                    continue

                self.presets_referencia[nome] = img_proc
                self.keypoints_presets[nome] = keypoints
                self.descritores_presets[nome] = descriptors
                logger.debug(f"Preset {nome}: {len(keypoints)} keypoints")

            except ValueError as e:
                logger.warning(f"Falha ao preprocessar preset '{nome}': {e}")
                continue

        logger.info(
            f"Configuração de presets concluída. Total válidos: {len(self.descritores_presets)} presets"
        )

    def _calcular_good_matches(
        self,
        descriptors_novos: np.ndarray,
        descriptors_preset: np.ndarray,
    ) -> int:
        """
        Realiza matching com Lowe ratio test.
        """
        if descriptors_novos is None or descriptors_preset is None:
            return 0

        if len(descriptors_novos) == 0 or len(descriptors_preset) == 0:
            return 0

        try:
            knn_matches = self.bf.knnMatch(descriptors_novos, descriptors_preset, k=2)
        except Exception as e:
            logger.warning(f"Falha no knnMatch: {e}")
            return 0

        ratio = float(self.good_match_ratio)
        good_matches = []
        for m_n in knn_matches:
            if len(m_n) < 2:
                continue
            m, n = m_n[0], m_n[1]
            if m.distance < ratio * n.distance:
                good_matches.append(m)

        return len(good_matches)

    def identificar_preset(
        self, imagem: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """
        Identifica o preset de uma nova imagem.

        Returns:
            (preset_id, score): score >= 0.
        """
        if not self.descritores_presets:
            logger.warning("Nenhum preset configurado; retornando None")
            return None, 0.0

        try:
            img_nova = self._preprocess_image(imagem)
        except ValueError as e:
            logger.error(f"Imagem inválida: {e}")
            raise

        keypoints_novos, descriptors_novos = self._extract_r2d2_features(img_nova)

        if descriptors_novos is None or len(keypoints_novos) == 0:
            logger.warning("Não foi possível extrair descritores da imagem nova")
            return None, 0.0

        logger.info(
            f"Identificando preset para imagem com {len(keypoints_novos)} keypoints"
        )

        matches_por_preset: Dict[str, int] = {}

        for nome_preset, descriptors_preset in self.descritores_presets.items():
            num_matches = self._calcular_good_matches(
                descriptors_novos,
                descriptors_preset,
            )
            matches_por_preset[nome_preset] = num_matches
            logger.debug(f"{nome_preset}: {num_matches} good matches")

        if not matches_por_preset:
            return None, 0.0

        melhor_preset = max(matches_por_preset, key=matches_por_preset.get)
        melhor_score = float(matches_por_preset[melhor_preset])

        logger.info(
            f"Melhor preset: {melhor_preset} com {melhor_score} good matches"
        )

        if melhor_score >= self.min_good_matches:
            print(f"Preset identificado: {melhor_preset} com score {melhor_score}")
            return str(melhor_preset), melhor_score
        else:
            print(
                f"Nenhum preset passou os limites: melhor foi {melhor_preset} "
                f"com score {melhor_score}"
            )
            return None, melhor_score


recognizer = PresetRecognizer()