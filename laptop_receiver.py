import json
import time
import threading
import numpy as np
import joblib
import paho.mqtt.client as mqtt
from pynput.mouse import Button, Controller as MouseController

CHANNEL_STREAM = "airmouse_pro/user_01/stream"
PROFILE_FILE = "airmouse_profile.pkl"

mouse = MouseController()

# Muat profil hasil latihan/kalibrasi jika sudah ada
try:
    profile = joblib.load(PROFILE_FILE)
    print(f" Profil kalibrasi '{PROFILE_FILE}' berhasil dimuat!")
except Exception:
    print(" Profil belum ditemukan, menggunakan parameter default rancangan.")
    profile = {
        "deadzone_deg": 1.8,
        "precision_speed_ref": 12.0,
        "fast_speed_ref": 65.0,
        "horiz_axis": "yawRate",
        "invert_x": -1.0,
        "invert_y": -1.0,
        "scroll_deadzone_deg": 3.0,
        "motion_classifier": None
    }

# Variabel State Machine di Receiver
state_mode = "MOVE"         # MOVE | SCROLL
left_button_held = False    # Menjaga state LEFT_DOWN agar tidak double-down
last_packet_time = time.time()
last_scroll_time = 0.0

# Smoothing residu sub-piksel agar gerakan lambat tetap halus (tidak patah-patah)
residual_x = 0.0
residual_y = 0.0

# Freeze singkat (120 ms) sesaat setelah tombol ditekan agar kursor tidak loncat
freeze_until = 0.0

def compute_adaptive_gain(speed, user_sens):
    """
    Adaptive Sensitivity (Bab 4.1):
    - Gerakan lambat = gain rendah untuk presisi tinggi (klik tombol kecil)
    - Gerakan cepat  = gain meningkat mulus agar kursor cepat melintasi layar
    """
    p_ref = profile["precision_speed_ref"]
    f_ref = profile["fast_speed_ref"]
    base_gain = 0.18 * user_sens

    if speed <= p_ref:
        return base_gain * 0.65
    elif speed >= f_ref:
        return base_gain * 1.85
    else:
        ratio = (speed - p_ref) / max(1.0, (f_ref - p_ref))
        return base_gain * (0.65 + ratio * 1.20)

def safe_release_left():
    global left_button_held
    if left_button_held:
        mouse.release(Button.left)
        left_button_held = False
        print("[AUTO-RELEASE] MOUSE_UP dikirim otomatis (mencegah tombol nyangkut).")

def watchdog_loop():
    """Memantau Heartbeat: jika sinyal hilang > 1.5 detik saat HOLD, otomatis MOUSE_UP."""
    while True:
        time.sleep(0.3)
        if left_button_held and (time.time() - last_packet_time > 1.5):
            safe_release_left()

def on_message(client, userdata, msg):
    global state_mode, left_button_held, last_packet_time, last_scroll_time
    global residual_x, residual_y, freeze_until

    last_packet_time = time.time()
    try:
        pkt = json.loads(msg.payload.decode())
        pkt_type = pkt.get("t")

        # ---------------------------------------------------------
        # 1. PERGANTIAN MODE (SCROLL_ON / SCROLL_OFF)
        # ---------------------------------------------------------
        if pkt_type == "MODE":
            mode_val = pkt.get("m")
            if mode_val == "SCROLL_ON":
                state_mode = "SCROLL"
                print(" [MODE] SCROLL AKTIF (Miringkan HP Depan/Belakang/Kiri/Kanan)")
            elif mode_val == "SCROLL_OFF":
                state_mode = "MOVE"
                print(" [MODE] Kembali ke MOVE")

        # ---------------------------------------------------------
        # 2. EVENT TOMBOL (CLICK, HOLD DOWN, RELEASE UP, RIGHT CLICK)
        # ---------------------------------------------------------
        elif pkt_type == "BTN":
            btn = pkt.get("b")
            action = pkt.get("a")

            if btn == "LEFT":
                if action == "CLICK":
                    mouse.click(Button.left, 1)
                    print(" [KLIK KIRI] Tap singkat")
                elif action == "DOWN" and not left_button_held:
                    mouse.press(Button.left)
                    left_button_held = True
                    freeze_until = time.time() + 0.10  # Stabilkan kursor 100ms saat mulai hold
                    print(" [HOLD / DRAG MULAI] Tombol kiri ditahan (MOUSE_DOWN)")
                elif action == "UP" and left_button_held:
                    mouse.release(Button.left)
                    left_button_held = False
                    print(" [RELEASE] Drag selesai (MOUSE_UP)")

            elif btn == "RIGHT" and action == "CLICK":
                mouse.click(Button.right, 1)
                print(" [KLIK KANAN]")

        # ---------------------------------------------------------
        # 3. MODE SCROLL BERBASIS TILT (Vertikal & Horizontal)
        # ---------------------------------------------------------
        elif pkt_type == "SCROLL" and state_mode == "SCROLL":
            now = time.time()
            # Batasi frekuensi injeksi scroll (maksimum ~10-12 step/detik sesuai Bab 4.2)
            if now - last_scroll_time < 0.09:
                return

            tilt_pitch = pkt.get("tiltPitch", 0.0) # Depan / Belakang
            tilt_roll  = pkt.get("tiltRoll", 0.0)  # Kiri / Kanan
            dz = profile["scroll_deadzone_deg"]    # Deadzone scroll = 3 derajat

            scroll_x, scroll_y = 0, 0

            # Scroll Vertikal (Tilt depan/belakang)
            if abs(tilt_pitch) > dz:
                #Kecepatan proporsional terhadap sudut tilt (1 s/d 3 notch)
                steps_y = int(np.clip((abs(tilt_pitch) - dz) / 6.0 + 1, 1, 3))
                scroll_y = -steps_y if tilt_pitch > 0 else steps_y

            # Scroll Horizontal (Tilt kiri/kanan)
            if abs(tilt_roll) > dz + 1.0:
                steps_x = int(np.clip((abs(tilt_roll) - dz) / 7.0 + 1, 1, 3))
                scroll_x = steps_x if tilt_roll > 0 else -steps_x

            if scroll_x != 0 or scroll_y != 0:
                mouse.scroll(scroll_x, scroll_y)
                last_scroll_time = now

        # ---------------------------------------------------------
        # 4. MODE GERAK KURSOR (MOVE & DRAG)
        # ---------------------------------------------------------
        elif pkt_type == "MOVE" and state_mode != "SCROLL":
            if time.time() < freeze_until:
                return

            horiz_rate = pkt.get(profile["horiz_axis"], 0.0)
            vert_rate  = pkt.get("pitchRate", 0.0)
            user_sens  = pkt.get("sens", 1.5)

            speed = np.hypot(horiz_rate, vert_rate)

            # Deadzone Filtering agar kursor tidak bergetar saat tangan diam
            if speed < profile["deadzone_deg"]:
                return

            # Jika classifier AI dari proses training tersedia, cek apakah ini hanya tremor
            clf = profile.get("motion_classifier")
            if clf is not None:
                regime = clf.predict([[speed]])[0]
                if regime == 0 and speed < profile["deadzone_deg"] * 1.25:
                    return

            gain = compute_adaptive_gain(speed, user_sens)

            raw_dx = horiz_rate * gain * profile["invert_x"] + residual_x
            raw_dy = vert_rate  * gain * profile["invert_y"] + residual_y

            move_dx = int(raw_dx)
            move_dy = int(raw_dy)

            residual_x = raw_dx - move_dx
            residual_y = raw_dy - move_dy

            if move_dx != 0 or move_dy != 0:
                mouse.move(move_dx, move_dy)

    except Exception as e:
        print("Error:", e)

def main():
    # Jalankan thread Watchdog Auto-Release di latar belakang
    threading.Thread(target=watchdog_loop, daemon=True).start()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = on_message
    client.connect("broker.hivemq.com", 1883, 60)
    client.subscribe(CHANNEL_STREAM)

    print("=========================================================")
    print("   AIR MOUSE EVERYWHERE - RECEIVER (PC/Laptop) AKTIF")
    print("   Mendengarkan kanal:", CHANNEL_STREAM)
    print("   Fitur: Move (Adaptive) | Tilt-Scroll | Hold & Drag")
    print("=========================================================")
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        safe_release_left()
        print("\nReceiver dihentikan dengan aman.")

if __name__ == "__main__":
    main()