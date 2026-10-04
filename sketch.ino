#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "model.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define LDR_PIN 34
#define LED_PIN 2
#define OLED_SDA 21
#define OLED_SCL 22

const char* WIFI_SSID = "Wokwi-GUEST";
const char* WIFI_PASSWORD = "";

WebServer server(80);
Adafruit_SSD1306 display(128, 64, &Wire, -1);

constexpr int ARENA_SIZE = 12 * 1024;
uint8_t tensor_arena[ARENA_SIZE];

struct LogEntry {
  unsigned long ms;
  int raw;
  float input;
  int classId;
};

constexpr int MAX_LOGS = 20;
LogEntry logs[MAX_LOGS];
int logCount = 0;

const char* labels[] = {"Dark", "Normal", "Bright"};
int currentRaw = 0;
float currentInput = 0.0f;
int currentClass = 0;

const char DASHBOARD[] PROGMEM = R"rawliteral(
<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Smart Light Classifier</title>
<style>
body{font-family:Arial;margin:0;padding:20px;background:#f2f4f7;color:#222}
.card{max-width:850px;margin:auto;background:white;padding:22px;border-radius:14px;box-shadow:0 2px 10px #bbb}
h1{margin-top:0}.value{font-size:42px;font-weight:bold;margin:10px 0}
.badge{display:inline-block;padding:8px 14px;border-radius:20px;background:#ddd}
table{width:100%;border-collapse:collapse;margin-top:20px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}
</style></head><body><div class="card">
<h1>Smart Light Classifier</h1>
<div>LDR reading</div><div class="value" id="ldr">--</div>
<div>TinyML result: <span class="badge" id="cls">--</span></div>
<p>Normalized input: <span id="input">--</span> | LED: <span id="led">--</span></p>
<h2>Recent log</h2><table><thead><tr><th>Time</th><th>LDR</th><th>Input</th><th>Class</th></tr></thead><tbody id="rows"></tbody></table>
</div>
<script>
async function refresh(){
 const d=await fetch('/data').then(r=>r.json());
 document.getElementById('ldr').textContent=d.ldr;
 document.getElementById('cls').textContent=d.class;
 document.getElementById('input').textContent=d.input.toFixed(3);
 document.getElementById('led').textContent=d.led?'ON':'OFF';
 const log=await fetch('/log').then(r=>r.json());
 document.getElementById('rows').innerHTML=log.map(x=>'<tr><td>'+Math.round(x.ms/1000)+'s</td><td>'+x.raw+'</td><td>'+x.input.toFixed(3)+'</td><td>'+x.class+'</td></tr>').join('');
}
setInterval(refresh,1000);refresh();
</script></body></html>
)rawliteral";

void addLog(int raw, float input, int classId) {
  if (logCount < MAX_LOGS) {
    logs[logCount++] = {millis(), raw, input, classId};
  } else {
    for (int i = 1; i < MAX_LOGS; ++i) logs[i - 1] = logs[i];
    logs[MAX_LOGS - 1] = {millis(), raw, input, classId};
  }
}

void handleRoot() { server.send(200, "text/html", DASHBOARD); }

void handleData() {
  String json = "{"ldr":" + String(currentRaw) +
                ","input":" + String(currentInput, 3) +
                ","class":"" + labels[currentClass] +
                "","led":" + String(currentClass == 0 ? "true" : "false") + "}";
  server.send(200, "application/json", json);
}

void handleLog() {
  String json = "[";
  for (int i = 0; i < logCount; ++i) {
    if (i) json += ",";
    json += "{"ms":" + String(logs[i].ms) +
            ","raw":" + String(logs[i].raw) +
            ","input":" + String(logs[i].input, 3) +
            ","class":"" + labels[logs[i].classId] + ""}";
  }
  json += "]";
  server.send(200, "application/json", json);
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  analogReadResolution(12);
  Wire.begin(OLED_SDA, OLED_SCL);

  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("OLED init failed");
    while (true) delay(1000);
  }

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Dashboard: http://");
  Serial.println(WiFi.localIP());

  server.on("/", handleRoot);
  server.on("/data", handleData);
  server.on("/log", handleLog);
  server.begin();

  const tflite::Model* model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("TFLite schema mismatch");
    while (true) delay(1000);
  }

  static tflite::MicroMutableOpResolver<5> resolver;
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddRelu();
  resolver.AddReshape();
  resolver.AddQuantize();

  static tflite::MicroInterpreter interpreter(
      model, resolver, tensor_arena, ARENA_SIZE);

  if (interpreter.AllocateTensors() != kTfLiteOk) {
    Serial.println("Tensor allocation failed");
    while (true) delay(1000);
  }

  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);

  while (true) {
    server.handleClient();

    currentRaw = analogRead(LDR_PIN);
    currentInput = currentRaw / 4095.0f;
    input->data.f[0] = currentInput;

    if (interpreter.Invoke() == kTfLiteOk) {
      currentClass = 0;
      for (int i = 1; i < 3; ++i) {
        if (output->data.f[i] > output->data.f[currentClass]) currentClass = i;
      }

      digitalWrite(LED_PIN, currentClass == 0 ? HIGH : LOW);
      addLog(currentRaw, currentInput, currentClass);

      Serial.printf("LDR=%d input=%.3f class=%s LED=%s\n",
                    currentRaw, currentInput, labels[currentClass],
                    currentClass == 0 ? "ON" : "OFF");

      display.clearDisplay();
      display.setTextColor(SSD1306_WHITE);
      display.setTextSize(1);
      display.setCursor(0,0);
      display.println("SMART LIGHT + TinyML");
      display.setTextSize(2);
      display.setCursor(0,16);
      display.println(labels[currentClass]);
      display.setTextSize(1);
      display.setCursor(0,40);
      display.print("LDR: "); display.println(currentRaw);
      display.print("LED: "); display.println(currentClass == 0 ? "ON" : "OFF");
      display.setCursor(0,56);
      display.print(WiFi.localIP());
      display.display();
    }

    delay(1000);
  }
}
