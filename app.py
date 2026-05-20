import os
import json
import base64
from datetime import datetime, timezone

from google import genai
from google.genai import types
from flask import Flask, jsonify, render_template, request

# Flask 앱 생성
app = Flask(__name__)

# Gemini API 설정 (Vercel 환경변수 사용)
API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_SENSOR_ANALYSIS_INTERVAL_MIN = 5  # 보관소 특성에 맞춰 기본 주기 5분으로 조정

# 최신 센서 데이터를 메모리에 보관
_latest: dict = {}
_latest_image: dict = {}
_settings: dict = {
    "sensor_analysis_interval_min": DEFAULT_SENSOR_ANALYSIS_INTERVAL_MIN,
}


def _has_api_key() -> bool:
    return bool(API_KEY and API_KEY.strip())


def _friendly_gemini_error(err) -> str:
    msg = str(err or "").lower()
    if "api_key_invalid" in msg or "api key not found" in msg:
        return "Gemini API 키가 없거나 잘못되었습니다"
    if "invalid_argument" in msg:
        return "Gemini 요청 형식이 올바르지 않습니다"
    if "429" in msg or "resource_exhausted" in msg or "rate" in msg:
        return "Gemini 요청 한도를 초과했습니다"
    if "timeout" in msg:
        return "Gemini 응답 시간이 초과되었습니다"
    return "Gemini 호출 실패"


def _to_int(value, default_value: int) -> int:
    try:
        return int(value)
    except Exception:
        return default_value


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _due_seconds(last_iso: str | None, interval_sec: int) -> int:
    if not last_iso:
        return 0
    try:
        last_dt = datetime.fromisoformat(last_iso)
    except Exception:
        return 0
    elapsed = (datetime.now(timezone.utc) - last_dt).total_seconds()
    left = interval_sec - int(elapsed)
    return max(0, left)


def _decide_motors(temp: float, hum: float) -> tuple[bool, bool]:
    # 프리미엄 주류 보관 규칙: 
    # 고온(22°C 이상)이면 변질 방지를 위해 송풍 공조기(fan) ON
    # 저습(55% 이하)이면 코르크 건조 방지를 위해 가습 펌프(pump) ON
    fan_on = temp >= 22.0
    pump_on = hum <= 55.0
    return fan_on, pump_on


def _build_fallback_result(temp: float, hum: float, reason: str) -> str:
    fan_on, pump_on = _decide_motors(temp, hum)
    advice = "온도안정, 코르크보호"
    if len(advice) > 20:
        advice = advice[:20]
    return json.dumps(
        {
            "fan": "ON" if fan_on else "OFF",
            "pump": "ON" if pump_on else "OFF",
            "advice": advice,
            "reason": (reason or "Gemini 호출 실패")[:80],
        },
        ensure_ascii=False,
    )


def _call_gemini(prompt: str) -> str:
    client = genai.Client(api_key=API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    if not response.text:
        raise ValueError("Gemini response text is empty")
    return response.text


def _call_gemini_with_image(prompt: str, image_bytes: bytes, mime_type: str) -> str:
    client = genai.Client(api_key=API_KEY)
    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[prompt, image_part],
    )
    if not response.text:
        raise ValueError("Gemini image response text is empty")
    return response.text


def _build_image_prompt(question: str, temp, hum) -> str:
    q = (question or "현재 보관 중인 주류의 상태와 진열 환경을 분석해줘").strip()
    sensor_hint = ""
    if temp is not None and hum is not None:
        sensor_hint = f" 참고 셀러 센서값: 내온={temp}°C, 내습={hum}%."
    return (
        "당신은 프리미엄 위스키와 와인을 관리하는 전문 소물리에이자 주류 보관 마스터입니다. "
        "업로드된 스마트 셀러 내부의 주류 사진을 보고 사용자의 질문에 전문적으로 답하세요. "
        "주류 라벨의 가독성, 코르크의 밀봉 상태, 빛 노출이나 침전물 발생 위험 등을 종합 고려하세요. "
        f"질문: {q}."
        f"{sensor_hint}"
    )


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api/latest", methods=["GET"])
def api_latest():
    payload = dict(_latest)
    interval_min = _to_int(_settings.get("sensor_analysis_interval_min"), DEFAULT_SENSOR_ANALYSIS_INTERVAL_MIN)
    payload.update(
        {
            "sensor_analysis_interval_min": interval_min,
            "analysis_due_in_sec": _due_seconds(_latest.get("analysis_timestamp"), interval_min * 60),
        }
    )
    return jsonify(payload)


@app.route("/api/latest-image", methods=["GET"])
def api_latest_image():
    return jsonify(_latest_image)


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "GET":
        return jsonify(
            {
                "sensor_analysis_interval_min": _to_int(
                    _settings.get("sensor_analysis_interval_min"),
                    DEFAULT_SENSOR_ANALYSIS_INTERVAL_MIN,
                ),
                "min": 1,
                "max": 120,
            }
        )

    body = request.get_json(silent=True) or {}
    interval_min = _to_int(body.get("sensor_analysis_interval_min"), DEFAULT_SENSOR_ANALYSIS_INTERVAL_MIN)
    interval_min = max(1, min(120, interval_min))
    _settings["sensor_analysis_interval_min"] = interval_min
    return jsonify({"ok": True, "sensor_analysis_interval_min": interval_min})


@app.route("/favicon.ico", methods=["GET"])
def favicon_ico():
    return "", 204


@app.route("/favicon.png", methods=["GET"])
def favicon_png():
    return "", 204


# (구) 헬스체크 엔드포인트 — 하위 호환용
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/sensor", methods=["POST"])
def sensor():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "JSON body is required"}), 400

        temp = data.get("temperature")
        hum = data.get("humidity")
        if temp is None or hum is None:
            return jsonify({"error": "temperature and humidity are required"}), 400

        temp_f = float(temp)
        hum_f = float(hum)
        now_iso = _now_iso()

        # 최신 센서값은 항상 갱신
        _latest.update(
            {
                "temperature": temp_f,
                "humidity": hum_f,
                "timestamp": now_iso,
            }
        )

        print(f"[보관소 센서] 온도: {temp_f}, 습도: {hum_f}")

        interval_min = _to_int(_settings.get("sensor_analysis_interval_min"), DEFAULT_SENSOR_ANALYSIS_INTERVAL_MIN)
        due_in_sec = _due_seconds(_latest.get("analysis_timestamp"), interval_min * 60)

        if due_in_sec > 0 and _latest.get("result"):
            return jsonify(
                {
                    "result": _latest.get("result"),
                    "analysis_run": False,
                    "analysis_reason": "not_due",
                    "analysis_due_in_sec": due_in_sec,
                    "sensor_analysis_interval_min": interval_min,
                }
            )

        prompt = (
            "당신은 스마트 주류 보관소(위스키 및 와인 셀러)의 인버터 환경 공조 제어 시스템입니다. "
            "아래 규칙을 반드시 지켜 단 한 줄의 JSON 블록만 출력하세요. "
            "마크다운 코드블록(```), 백틱, 불필요한 공백, 추가 설명이나 개행은 절대 금지합니다. "
            f"현재 계측 데이터: temperature={temp_f}, humidity={hum_f}. "
            "출력 JSON 스키마: {\"fan\":\"ON|OFF\",\"pump\":\"ON|OFF\",\"advice\":\"20자 이내\"}. "
            "위스키 숙성 가스 정체 및 와인 고온 변질 방지를 판단하여 fan(송풍기)을 제어하고, 코르크 건조 방지를 위해 pump(가습기)를 제어하세요. "
            "advice는 보관 최적화 팁을 한국어 20자 이내로 간결하게 작성하세요."
        )

        result = ""
        ai_error = None

        if _has_api_key():
            try:
                result = _call_gemini(prompt)
            except Exception as e:
                ai_error = _friendly_gemini_error(e)
        else:
            ai_error = "Gemini API 키가 설정되지 않았습니다"

        if not result:
            result = _build_fallback_result(temp_f, hum_f, ai_error or "unknown")

        print(f"[AI 분석 응답] {result}")

        # 최신 데이터 저장 (대시보드 표시용)
        _latest.update(
            {
                "result": result,
                "analysis_timestamp": now_iso,
            }
        )

        response = {
            "result": result,
            "analysis_run": True,
            "analysis_reason": "ok" if not ai_error else "fallback",
            "analysis_due_in_sec": _due_seconds(_latest.get("analysis_timestamp"), interval_min * 60),
            "sensor_analysis_interval_min": interval_min,
        }
        if ai_error:
            response["fallback"] = True
            response["detail"] = ai_error
        return jsonify(response)

    except Exception as e:
        print("에러:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/plant-image", methods=["POST"])
def plant_image():
    # ESP32-CAM 등 하드웨어 펌웨어 소스 코드의 엔드포인트명(/plant-image) 하위 호환 유지
    try:
        if "image" not in request.files:
            return jsonify({"error": "multipart form-data field 'image' is required"}), 400

        file = request.files["image"]
        if not file or not file.filename:
            return jsonify({"error": "image file is empty"}), 400

        image_bytes = file.read()
        if not image_bytes:
            return jsonify({"error": "image file has no bytes"}), 400
        if len(image_bytes) > 4 * 1024 * 1024:
            return jsonify({"error": "image is too large (max 4MB)"}), 400

        mime_type = file.mimetype or "image/jpeg"
        if not mime_type.startswith("image/"):
            return jsonify({"error": "uploaded file must be an image"}), 400

        now_iso = datetime.now(timezone.utc).isoformat()
        _latest.update({"plant_timestamp": now_iso})

        _latest_image.update(
            {
                "timestamp": now_iso,
                "mime_type": mime_type,
                "image_b64": base64.b64encode(image_bytes).decode("ascii"),
            }
        )

        return jsonify({"ok": True, "timestamp": now_iso, "message": "image_saved"})

    except Exception as e:
        print("[cellar-image] 에러:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyze-image", methods=["POST"])
def analyze_image():
    try:
        if "image_b64" not in _latest_image or "mime_type" not in _latest_image:
            return jsonify({"error": "latest cellar image not found"}), 400

        body = request.get_json(silent=True) or {}
        question = body.get("question", "")

        if not _has_api_key():
            return jsonify({"error": "Gemini API 키가 설정되지 않았습니다"}), 500

        image_bytes = base64.b64decode(_latest_image["image_b64"])
        prompt = _build_image_prompt(question, _latest.get("temperature"), _latest.get("humidity"))
        try:
            result = _call_gemini_with_image(prompt, image_bytes, _latest_image["mime_type"])
        except Exception as e:
            return jsonify({"error": _friendly_gemini_error(e)}), 500

        now_iso = datetime.now(timezone.utc).isoformat()
        _latest.update(
            {
                "plant_result": result,
                "plant_timestamp": now_iso,
                "plant_question": question,
            }
        )
        _latest_image.update({"plant_result": result, "plant_timestamp": now_iso})

        return jsonify({"ok": True, "result": result, "timestamp": now_iso})

    except Exception as e:
        print("[analyze-image] 에러:", e)
        return jsonify({"error": "이미지 분석 처리 중 서버 오류"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)