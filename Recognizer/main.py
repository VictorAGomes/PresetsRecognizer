import json
import time
from pathlib import Path

import cv2
import numpy as np
from algorithms.akaze import PresetRecognizer as AKAZERecognizer
from algorithms.alike_preset import AlikePresetRecognizer
from algorithms.brisk import PresetRecognizer as BRISKRecognizer
from algorithms.disk_recognizer import PresetRecognizer as DiskRecognizer
from algorithms.kaze import PresetRecognizer as KAZERecognizer

# Importação dos algoritmos disponíveis
from algorithms.orb import PresetRecognizer as ORBRecognizer
from algorithms.r2d2_preset import PresetRecognizer as R2D2Recognizer
from algorithms.sift import PresetRecognizer as SIFTRecognizer
from algorithms.superpoint_recognizer import (
    SuperPointPresetRecognizer as SuperPointRecognizer,
)

PROJECT_ROOT = Path(__file__).resolve().parent
PRESET_TEST_JSON = PROJECT_ROOT / "data/cameras.json"


def get_recognizer(algorithm: str):
    if algorithm == "orb":
        return ORBRecognizer(min_good_matches=1)
    elif algorithm == "sift":
        return SIFTRecognizer(min_good_matches=1)
    elif algorithm == "brisk":
        return BRISKRecognizer(min_good_matches=1)
    elif algorithm == "kaze":
        return KAZERecognizer(min_good_matches=1)
    elif algorithm == "akaze":
        return AKAZERecognizer(min_good_matches=1)
    elif algorithm == "r2d2":
        return R2D2Recognizer(min_good_matches=1)
    elif algorithm == "superpoint":
        return SuperPointRecognizer(min_good_matches=1)
    elif algorithm == "disk":
        return DiskRecognizer(min_good_matches=1)
    elif algorithm == "alike":
        return AlikePresetRecognizer(
            model_name="alike-t",
            model_weights_path="algorithms/alike/alike-t.pth",
            min_good_matches=1,
        )
    else:
        raise ValueError(f"Algoritmo desconhecido: {algorithm}")


def load_image(path: str) -> np.ndarray:
    if path.startswith("/"):
        project_root = Path(__file__).resolve().parent
        abs_path = project_root / path[1:]
        img = cv2.imread(str(abs_path))
    else:
        img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(
            f"Image not found: {abs_path if path.startswith('/') else path}"
        )
    return img


def main(algorithm: str = "orb"):
    with open(PRESET_TEST_JSON, "r", encoding="utf-8") as f:
        spec = json.load(f)

    all_results = []
    total_tests = 0
    total_correct = 0
    total_score = 0.0
    wrong_presets = []
    start_time = time.time()

    for cam in spec.get("cameras", []):
        camera_id = cam.get("camera_id")
        presets = {
            str(p["id"]): load_image(p["image_path"]) for p in cam.get("presets", [])
        }
        recognizer = get_recognizer(algorithm)
        recognizer.configurar_presets(presets)

        for test in cam.get("tests", []):
            img = load_image(test["image_path"])
            expected = str(test["expected_preset"])
            # Calcula tempo de processamento
            start_test_time = time.time()
            detected, score = recognizer.identificar_preset(img)
            end_test_time = time.time()
            processing_time = end_test_time - start_test_time
            correct = detected == expected
            result = {
                "camera_id": camera_id,
                "expected": expected,
                "detected": detected,
                "score": score,
                "correct": correct,
                "test_image": test["image_path"],
                "processing_time_sec": processing_time,
            }
            all_results.append(result)
            total_tests += 1
            total_score += score
            if correct:
                total_correct += 1
            else:
                wrong_presets.append(result)

    elapsed_time = time.time() - start_time
    accuracy = total_correct / total_tests if total_tests > 0 else 0.0
    mean_score = total_score / total_tests if total_tests > 0 else 0.0

    summary = {
        "algorithm": algorithm,
        "accuracy": accuracy,
        "mean_score": mean_score,
        "elapsed_time_sec": elapsed_time,
        "total_tests": total_tests,
        "total_correct": total_correct,
        "wrong_presets": wrong_presets,
        "results": all_results,
    }

    datetime_now = time.strftime("%Y%m%d-%H%M%S")
    RESULTS_JSON = (
        PROJECT_ROOT / f"resultados/resultados-{datetime_now}-{algorithm}.json"
    )

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Acurácia: {accuracy:.2%}")
    print(f"Média de score: {mean_score:.2f}")
    print(f"Tempo de execução: {elapsed_time:.2f} segundos")
    print(f"Total de testes: {total_tests}")
    print(f"Total corretos: {total_correct}")
    print(f"Presets errados: {len(wrong_presets)} (detalhes em {RESULTS_JSON})")


if __name__ == "__main__":
    for i in range(1):
        for alg in [
            "alike",
            "sift",
            "brisk",
            "kaze",
            "akaze",
            "r2d2",
            "superpoint",
            "alike",
        ]:
            print(f"\n=== Executando com o algoritmo: {alg} ===")
            main(alg)
