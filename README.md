# Mini Integration Project — Smart Light Classifier

An ESP32 smart-light system combining **WiFi, LDR sensing, TinyML, OLED display, automatic LED control, and a web dashboard**.

## Features
- Reads a live LDR value on GPIO 34.
- Normalizes the sensor reading and classifies it locally with TensorFlow Lite Micro.
- Classes: **Dark, Normal, Bright**.
- Shows classification and sensor value on an SSD1306 OLED.
- Automatically turns the LED ON when the TinyML result is Dark.
- Hosts a WiFi dashboard showing the live result and recent logged readings.
- Dashboard refreshes every second and displays the last 20 measurements.

## Connections
| Component | ESP32 |
|---|---|
| LDR AO | GPIO 34 |
| LDR VCC | 3.3V |
| LDR GND | GND |
| OLED SDA | GPIO 21 |
| OLED SCL | GPIO 22 |
| OLED VCC | 3.3V |
| OLED GND | GND |
| LED anode | GPIO 2 through resistor |
| LED cathode | GND |

## WiFi
For Wokwi:
- SSID: `Wokwi-GUEST`
- Password: empty
- Dashboard: open the IP printed in Serial Monitor.

For real hardware, change the SSID/password in `sketch.ino`.

## TinyML model
The model expects one float input from 0.0 to 1.0:
- 0 = Dark
- 1 = Normal
- 2 = Bright

Run `Smart_Light_TinyML.ipynb` to train/export `light_level_classifier.tflite`, then run `convert_model.py` to generate the real `model.h`.

**Important:** the committed `model.h` is a template until the generated TFLite model is converted.

## Dashboard
The ESP32 serves:
- `/` — live dashboard
- `/data` — current JSON reading
- `/log` — recent JSON log

No cloud service is required; logging is kept in ESP32 RAM.
