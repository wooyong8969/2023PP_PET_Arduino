char cmd;

void setup() {
  Serial.begin(9600);
  pinMode(9, OUTPUT);
}

void loop() {
  if(Serial.available()){
    cmd = Serial.read();
    Serial.println(cmd);
    if(cmd == 'a') digitalWrite(9, HIGH);
    else digitalWrite(9, LOW);
  }
}
