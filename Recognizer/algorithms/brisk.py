import datetime
import cv2
import os
import numpy as np
from typing import Dict, Tuple, Optional, List
import base64
import logging
import io


import cv2

N_FEATURES = 10000
THRESH = 30
OCTAVES = 3
PATTERN_SCALE = 1.0

logger = logging.getLogger(__name__)

class PresetRecognizer:
    """
    Classe para reconhecimento de presets de câmera usando OpenCV e feature matching.
    Pipeline: pré-processamento -> BRISK -> Lowe ratio test
    """

    def __init__(
        self,
        num_features: int = 8000,
        good_match_ratio: float = 0.75,
        min_good_matches: int = 10,
        target_size: Tuple[int, int] = (640, 480),
    ) -> None:
        """
        Inicializa o reconhecedor de presets.

        Args:
            num_features: número máximo de features BRISK por imagem.
            good_match_ratio: razão de Lowe no matching (0 < ratio < 1).
            target_size: tamanho para redimensionar imagens (largura, altura).
        """
        # validação básica de parâmetros
        if not (0 < good_match_ratio < 1):
            raise ValueError("good_match_ratio deve estar em (0, 1).")

        self.num_features = num_features
        self.good_match_ratio = good_match_ratio
        self.min_good_matches = min_good_matches
        self.target_size = target_size

        # Detector BRISK (parâmetros ajustados)
        self.brisk = cv2.BRISK_create(
            thresh=THRESH, octaves=OCTAVES, patternScale=PATTERN_SCALE
        )

        # Matcher
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Descritores dos presets de referência
        self.descritores_presets: Dict[str, Tuple[List[cv2.KeyPoint], np.ndarray]] = {}

        # Imagens de referência
        self.presets_referencia: Dict[str, np.ndarray] = {}

    def _preprocess_image(self, imagem: np.ndarray) -> np.ndarray:
        """
        Converte para grayscale (se necessário), redimensiona para target_size
        e aplica opcionalmente correção gamma e CLAHE para reduzir sensibilidade à iluminação.
        """
        if imagem is None:
            raise ValueError("Imagem inválida: None")

        # converte para grayscale
        if len(imagem.shape) == 3:
            # assume BGR (padrão OpenCV)
            imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

        try:
            imagem = cv2.resize(imagem, self.target_size)
        except cv2.error as e:
            raise ValueError(
                f"Falha ao redimensionar imagem de tamanho {imagem.shape}: {e}"
            )

        img_proc = imagem.copy()

        return img_proc

    def _compute_descriptors(
        self, imagem: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """
        Detecta keypoints e descritores BRISK de forma segura.
        """
        if imagem is None:
            return [], None
        try:
            keypoints, descriptors = self.brisk.detectAndCompute(imagem, None)
        except Exception as e:
            logger.warning(f"Falha ao detectar/calcular descritores BRISK: {e}")
            return [], None
        if descriptors is None or len(keypoints) == 0:
            return [], None

        return keypoints, descriptors

    # Configuração de presets
    def configurar_presets(self, presets: Dict[str, np.ndarray]) -> None:
        """
        Configura os presets de referência a partir de imagens em memória.
        As imagens são convertidas para grayscale, redimensionadas para target_size
        e passam pelo mesmo pré-processamento da imagem de entrada.
        """
        self.presets_referencia = {}
        for nome, img in presets.items():
            if img is None:
                logger.warning(f"Imagem do preset '{nome}' é None; ignorando")
                continue
            try:
                img_proc = self._preprocess_image(img)
            except ValueError as e:
                logger.warning(f"Falha ao preprocessar preset '{nome}': {e}")
                continue
            self.presets_referencia[nome] = img_proc
        self._computar_descritores_presets()

    def _computar_descritores_presets(self) -> None:
        """
        Computa descritores BRISK para todos os presets em self.presets_referencia.
        """
        self.descritores_presets = {}
        for nome_preset, imagem in self.presets_referencia.items():
            try:
                keypoints, descriptors = self._compute_descriptors(imagem)
                if descriptors is None or len(keypoints) == 0:
                    logger.warning(
                        f"Preset '{nome_preset}' não possui descritores suficientes; ignorando"
                    )
                    continue

                self.descritores_presets[nome_preset] = (keypoints, descriptors)
                logger.debug(f"Preset {nome_preset}: {len(keypoints)} keypoints")
            except Exception as e:
                logger.error(f"Erro ao processar preset '{nome_preset}': {e}")

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
        Retorna o número de good matches.
        """
        if descriptors_novos is None or descriptors_preset is None:
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

    # Identificação de preset
    def identificar_preset(self, imagem: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Identifica o preset de uma nova imagem.

        Returns:
            (preset_id, score): score >= 0.
            Se não identificado pelos critérios, retorna (None, melhor_score)
            onde melhor_score é o maior número de good matches entre os presets.
        """
        if not self.descritores_presets:
            logger.warning("Nenhum preset configurado; retornando None")
            return None, 0.0

        try:
            img_nova_gray = self._preprocess_image(imagem)
        except ValueError as e:
            logger.error(f"Imagem inválida ou não encontrada: {e}")
            raise

        keypoints_novos, descriptors_novos = self._compute_descriptors(img_nova_gray)
        if descriptors_novos is None or len(keypoints_novos) == 0:
            logger.warning("Não foi possível extrair descritores da imagem nova")
            return None, 0.0

        logger.info(
            f"Identificando preset para imagem com {len(keypoints_novos)} keypoints"
        )

        matches_por_preset: Dict[str, int] = {}

        for nome_preset, (keypoints_preset, descriptors_preset) in list(
            self.descritores_presets.items()
        ):
            if descriptors_preset is None or len(keypoints_preset) == 0:
                logger.debug(
                    f"{nome_preset}: ignorado (sem descritores válidos no preset)"
                )
                continue

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
            f"Melhor preset: {melhor_preset} " f"com {melhor_score} good matches"
        )

        if melhor_score >= self.min_good_matches:
            print(f"Preset identificado: {melhor_preset} " f"com score {melhor_score}")
            return str(melhor_preset), melhor_score
        else:
            print(
                f"Nenhum preset passou os limites: melhor foi {melhor_preset} "
                f"com score {melhor_score}"
            )
            return None, melhor_score


recognizer = PresetRecognizer()
