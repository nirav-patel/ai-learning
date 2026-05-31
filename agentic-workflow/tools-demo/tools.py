"""
Tool definitions for the tools demo.

Four tools that demonstrate different use cases:
  - get_current_time      : no-arg, real-time data
  - get_weather_from_ip   : no-arg, external API call
  - write_txt_file        : side-effect (file I/O)
  - generate_qr_code      : generates a PNG image
"""

import os
from datetime import datetime

import requests


# ---------------------------------------------------------------------------
# Tool 1 — Current time
# ---------------------------------------------------------------------------

def get_current_time() -> str:
    """
    Returns the current local time as a string.
    """
    return datetime.now().strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Tool 2 — Weather at current location (detected via IP)
# ---------------------------------------------------------------------------

def get_weather_from_ip() -> str:
    """
    Gets the current, high, and low temperature in Celsius for the user's
    location and returns it as a formatted string.
    """
    try:
        loc = requests.get("https://ipinfo.io/json", timeout=5).json()["loc"]
        lat, lon = loc.split(",")
    except Exception as exc:
        return f"Could not detect location: {exc}"

    params = {
        "latitude":         lat,
        "longitude":        lon,
        "current":          "temperature_2m",
        "daily":            "temperature_2m_max,temperature_2m_min",
        "temperature_unit": "celsius",
        "timezone":         "auto",
    }
    try:
        data = requests.get(
            "https://api.open-meteo.com/v1/forecast", params=params, timeout=5
        ).json()
        current = data["current"]["temperature_2m"]
        high    = data["daily"]["temperature_2m_max"][0]
        low     = data["daily"]["temperature_2m_min"][0]
        return f"Current: {current}°C, High: {high}°C, Low: {low}°C"
    except Exception as exc:
        return f"Could not fetch weather: {exc}"


# ---------------------------------------------------------------------------
# Tool 3 — Write a text file
# ---------------------------------------------------------------------------

def write_txt_file(file_path: str, content: str) -> str:
    """
    Write text content into a .txt file, creating parent directories if needed.
    Args:
        file_path: Destination path (e.g. output/reminders.txt).
        content: Text to write into the file.
    Returns:
        Confirmation string with the path written.
    """
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"File written: {file_path}"


# ---------------------------------------------------------------------------
# Tool 4 — Generate a QR code PNG
# ---------------------------------------------------------------------------

def generate_qr_code(data: str, filename: str) -> str:
    """
    Generate a QR code PNG image from a URL or text string.
    Args:
        data: The URL or text to encode in the QR code.
        filename: Output file path for the PNG (e.g. output/my_qr.png).
    Returns:
        Confirmation string with the output path.
    """
    try:
        import qrcode  # type: ignore
    except ImportError:
        return "qrcode package not installed. Run: pip install qrcode[pil]"

    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)

    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")

    output_path = filename if filename.endswith(".png") else f"{filename}.png"
    img.save(output_path)
    return f"QR code saved as {output_path} encoding: {data[:60]}"
