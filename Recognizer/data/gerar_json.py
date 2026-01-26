import os
import json
import re

TEST_ROOT = "Recognizer/data/images-test"
REF_ROOT = "Recognizer/data/images-ref"

# Câmeras que você quer exportar
CAMERAS = [
    "cam_14",
    "cam_15",
    "cam_30",
    "cam_33",
    "cam_37",
    "cam_43",
    "cam_48",
    "cam_57",
    "cam_58",
    "cam_78",
    "cam_82",
    "cam_88",
    "cam_95",
    "cam_106",
    "cam_131",
    "cam_133",
    "cam_136",
    "cam_138",
    "cam_151",
    "cam_157",
    "cam_163",
]

OUTPUT_JSON = "Recognizer/data/cameras.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PRESET_REGEX = re.compile(r"preset_(\d+)", re.IGNORECASE)
PRESET_IMG_REGEX = re.compile(
    r"Preset_([a-f0-9\-]+)_(\d+)(?:_(?!tarde$|noite$)(\w+))?\.(jpg|jpeg|png|bmp|webp)$",
    re.IGNORECASE
)


def relpath_from_project(full_path):
    # Caminho relativo a partir da raiz do projeto
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    rel = os.path.relpath(full_path, project_root)
    rel = rel.replace("\\", "/")
    # Remove prefixo 'Recognizer/' se existir
    if rel.startswith("Recognizer/data/"):
        rel = rel[len("Recognizer/"):]
    if not rel.startswith("data/"):
        raise ValueError(f"Caminho inesperado: {rel}")
    return "/" + rel


def generate_json(test_root: str, ref_root: str, cameras: list[str]):
    data = {"cameras": []}

    for cam in cameras:
        cam_path = os.path.join(test_root, cam)

        if not os.path.isdir(cam_path):
            print(f"⚠️ Pasta não encontrada: {cam_path}")
            continue

        tests = []
        presets_dict = {}

        for preset_folder in sorted(os.listdir(cam_path)):
            preset_path = os.path.join(cam_path, preset_folder)

            if not os.path.isdir(preset_path):
                continue

            match = PRESET_REGEX.match(preset_folder)
            if not match:
                continue

            preset_id = match.group(1)

            for filename in sorted(os.listdir(preset_path)):
                ext = os.path.splitext(filename)[1].lower()
                if ext not in IMAGE_EXTENSIONS:
                    continue

                image_path = os.path.join(test_root, cam, preset_folder, filename)
                image_path = relpath_from_project(image_path)  # <-- ajuste aqui

                tests.append({
                    "expected_preset": preset_id,
                    "image_path": image_path
                })

        # Agora busca os presets nas imagens de referência
        ref_cam_path = os.path.join(ref_root, cam)
        if os.path.isdir(ref_cam_path):
            for fname in sorted(os.listdir(ref_cam_path)):
                match = PRESET_IMG_REGEX.match(fname)
                if not match:
                    continue
                preset_id = match.group(2)
                image_path = os.path.join(ref_cam_path, fname)
                image_path = relpath_from_project(image_path)  # <-- ajuste aqui
                # Só adiciona se ainda não existe para esse preset_id
                if preset_id not in presets_dict:
                    presets_dict[preset_id] = image_path

        # Monta a lista de presets no formato desejado
        presets = [
            {"id": pid, "image_path": path}
            for pid, path in sorted(presets_dict.items(), key=lambda x: int(x[0]))
        ]

        camera_entry = {
            "camera_id": cam,
            "presets": presets,
            "tests": tests
        }

        data["cameras"].append(camera_entry)

        print(f"✅ {cam}: {len(tests)} imagens encontradas. {len(presets)} presets encontrados.")

    return data


if __name__ == "__main__":
    result = generate_json(TEST_ROOT, REF_ROOT, CAMERAS)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n📄 JSON gerado em: {OUTPUT_JSON}")
