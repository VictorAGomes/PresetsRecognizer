import logging
import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch

# Import the ALIKE model class
# Assuming the file structure: Recognizer/algorithms/alike/alike.py
from algorithms.alike.alike import ALike, configs

logger = logging.getLogger(__name__)


class AlikePresetRecognizer:
    """
    Classe para reconhecimento de presets de câmera usando ALIKE.
    Pipeline: pré-processamento -> ALIKE extração -> matching com descritores
    """

    def __init__(
        self,
        model_name: str = "alike-t",
        model_weights_path: Optional[str] = None,
        good_match_ratio: float = 0.75,
        min_good_matches: int = 10,
        target_size: Tuple[int, int] = (640, 480),
        top_k: int = 2000,
        scores_th: float = 0.2,
    ) -> None:
        """
        Inicializa o reconhecedor de presets com ALIKE.

        Args:
            model_name: variante do modelo ('alike-t', 'alike-s', 'alike-n', 'alike-l').
            model_weights_path: caminho para os pesos (.pth). Se None, tenta usar o padrão da config.
            good_match_ratio: razão de Lowe no matching (0 < ratio < 1).
            min_good_matches: número mínimo de good matches para identificar um preset.
            target_size: tamanho para redimensionar imagens (largura, altura).
            top_k: número máximo de keypoints.
            scores_th: threshold de score para keypoints.
        """
        if not (0 < good_match_ratio < 1):
            raise ValueError("good_match_ratio deve estar em (0, 1).")

        self.good_match_ratio = good_match_ratio
        self.min_good_matches = min_good_matches
        self.target_size = target_size
        self.top_k = top_k

        # Configuração do modelo ALIKE
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if model_name not in configs:
            raise ValueError(
                f"Modelo {model_name} desconhecido. Opções: {list(configs.keys())}"
            )

        cfg = configs[model_name]

        # Override path if provided
        weights_path = model_weights_path if model_weights_path else cfg["model_path"]
        # If the path is relative and we are running from Recognizer root, we might need to adjust
        if not os.path.exists(weights_path) and model_weights_path is None:
            # Fallback to look in relative path
            weights_path = os.path.join(
                "algorithms/alike", os.path.basename(cfg["model_path"])
            )

        logger.info(f"Carregando ALIKE ({model_name}) de: {weights_path}")

        self.model = ALike(
            c1=cfg["c1"],
            c2=cfg["c2"],
            c3=cfg["c3"],
            c4=cfg["c4"],
            dim=cfg["dim"],
            single_head=cfg["single_head"],
            radius=cfg["radius"],
            top_k=top_k,
            scores_th=scores_th,
            model_path=weights_path,
            device=self.device,
        )
        self.model.eval()

        # Matcher - ALIKE descriptors are float, so use L2 Norm
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        # Descritores dos presets
        self.descritores_presets: Dict[str, np.ndarray] = {}
        self.keypoints_presets: Dict[str, np.ndarray] = {}
        self.presets_referencia: Dict[str, np.ndarray] = {}

    def _preprocess_image(self, imagem: np.ndarray) -> np.ndarray:
        """
        Converte para RGB (se necessário) e redimensiona para target_size.
        ALIKE espera RGB.
        """
        if imagem is None:
            raise ValueError("Imagem inválida: None")

        # Converte BGR para RGB se necessário (OpenCV carrega em BGR)
        if len(imagem.shape) == 3 and imagem.shape[2] == 3:
            imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

        try:
            imagem = cv2.resize(imagem, self.target_size)
        except cv2.error as e:
            raise ValueError(
                f"Falha ao redimensionar imagem de tamanho {imagem.shape}: {e}"
            )

        return imagem

    def _extract_features(
        self, imagem: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extrai keypoints e descritores ALIKE.

        Returns:
            keypoints: (N, 2)
            descriptors: (N, D)
        """
        if imagem is None or imagem.size == 0:
            return np.array([]), None

        try:
            # ALIKE forward method requires RGB numpy array (H, W, 3)
            # Preprocessing already ensures RGB and Resize

            # Forward pass
            # Note: image_size_max is arbitrary large because we already resized
            with torch.no_grad():
                pred = self.model(imagem, sub_pixel=True)

            keypoints = pred["keypoints"]
            descriptors = pred["descriptors"]

            if len(keypoints) == 0:
                return np.array([]), None

            return keypoints, descriptors

        except Exception as e:
            logger.warning(f"Falha ao extrair features ALIKE: {e}")
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
                keypoints, descriptors = self._extract_features(img_proc)

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
            except Exception as e:
                logger.error(f"Erro ao configurar preset '{nome}': {e}")
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

    def identificar_preset(self, imagem: np.ndarray) -> Tuple[Optional[str], float]:
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
            return None, 0.0

        keypoints_novos, descriptors_novos = self._extract_features(img_nova)

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

        logger.info(f"Melhor preset: {melhor_preset} com {melhor_score} good matches")

        if melhor_score >= self.min_good_matches:
            print(f"Preset identificado: {melhor_preset} com score {melhor_score}")
            return str(melhor_preset), melhor_score
        else:
            print(
                f"Nenhum preset passou os limites: melhor foi {melhor_preset} "
                f"com score {melhor_score}"
            )
            return None, melhor_score
