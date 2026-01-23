import os
import json
import cv2
import time
from pathlib import Path
import numpy as np

# Importação dos algoritmos disponíveis
from algorithms.orb import PresetRecognizer as ORBRecognizer
from algorithms.sift import PresetRecognizer as SIFTRecognizer
from algorithms.brisk import PresetRecognizer as BRISKRecognizer
from algorithms.r2d2_preset import PresetRecognizer as R2D2Recognizer
from algorithms.superpoint_recognizer import SuperPointPresetRecognizer as SuperPointRecognizer

datetime_now = time.strftime("%Y%m%d-%H%M%S")
PROJECT_ROOT = Path(__file__).resolve().parent
PRESET_TEST_JSON = PROJECT_ROOT / "data/cameras.json"
RESULTS_JSON = PROJECT_ROOT / f"resultados/resultados-{datetime_now}.json"

# Escolha do algoritmo: "orb" ou "sift" ou "brisk" ou "r2d2" ou "superpoint"
ALGORITHM = "superpoint"

def get_recognizer(algorithm: str):
    if algorithm == "orb":
        return ORBRecognizer()
    elif algorithm == "sift":
        return SIFTRecognizer()
    elif algorithm == "brisk":
        return BRISKRecognizer()
    elif algorithm == "r2d2":
        return R2D2Recognizer()
    elif algorithm == "superpoint":
        return SuperPointRecognizer()
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
        raise FileNotFoundError(f"Image not found: {abs_path if path.startswith('/') else path}")
    return img


def main():
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
        recognizer = get_recognizer(ALGORITHM)
        recognizer.configurar_presets(presets)

        for test in cam.get("tests", []):
            img = load_image(test["image_path"])
            expected = str(test["expected_preset"])
            detected, score = recognizer.identificar_preset(img)
            correct = detected == expected
            result = {
                "camera_id": camera_id,
                "expected": expected,
                "detected": detected,
                "score": score,
                "correct": correct,
                "test_image": test["image_path"]
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
        "algorithm": ALGORITHM,
        "accuracy": accuracy,
        "mean_score": mean_score,
        "elapsed_time_sec": elapsed_time,
        "total_tests": total_tests,
        "total_correct": total_correct,
        "wrong_presets": wrong_presets,
        "results": all_results
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Acurácia: {accuracy:.2%}")
    print(f"Média de score: {mean_score:.2f}")
    print(f"Tempo de execução: {elapsed_time:.2f} segundos")
    print(f"Total de testes: {total_tests}")
    print(f"Total corretos: {total_correct}")
    print(f"Presets errados: {len(wrong_presets)} (detalhes em {RESULTS_JSON})")


if __name__ == "__main__":
    main()