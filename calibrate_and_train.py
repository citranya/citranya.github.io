import json
import time
import numpy as np
import joblib
import paho.mqtt.client as mqtt
from sklearn.neighbors import KNeighborsClassifier

CHANNEL_STREAM = "airmouse_pro/user_01/stream"
PROFILE_FILE = "airmouse_profile.pkl"

samples = []
is_recording = False

def on_message(client, userdata, msg):
    global samples, is_recording
    if not is_recording:
        return
    try:
        data = json.loads(msg.payload.decode())
        if data.get("t") in ["MOVE", "SCROLL"]:
            samples.append(data)
    except Exception:
        pass

def collect_for_seconds(duration, prompt_text):
    global samples, is_recording
    input(f"\n Tekan [ENTER] untuk mulai: {prompt_text}")
    print("   Siap-siap dalam 2 detik...")
    time.sleep(2.0)
    samples.clear()
    is_recording = True
    print(f"    MEREKAM SELAMA {duration} DETIK... LAKUKAN SEKARANG!")
    time.sleep(duration)
    is_recording = False
    print(f"    Selesai! ({len(samples)} paket data terekam)")
    return list(samples)

def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = on_message
    client.connect("broker.hivemq.com", 1883, 60)
    client.subscribe(CHANNEL_STREAM)
    client.loop_start()

    print("=========================================================")
    print(" KALIBRASI & TRAINING PROFIL AIR MOUSE PERSONAL")
    print("=========================================================")
    print("Pastikan halaman index.html di HP sudah dibuka dan tombol")
    print("'AKTIFKAN SENSOR' sudah ditekan.")

    # -------------------------------------------------------------
    # TAHAP 1: Rekam Getaran Tangan Diam & Saat Tap (Deadzone & Anti-Click)
    # -------------------------------------------------------------
    still_data = collect_for_seconds(
        4, "TAHAP 1 - Pegang HP diam mengarah ke layar sambil sesekali menekan jari di layar (4 detik)"
    )
    still_speeds = [np.hypot(d.get("yawRate", 0), d.get("pitchRate", 0)) for d in still_data if "yawRate" in d]
    # Deadzone otomatis = persentil ke-95 dari getaran tangan saat diam + sedikit margin
    deadzone_deg = float(np.percentile(still_speeds, 95) * 1.15) if still_speeds else 2.0
    deadzone_deg = max(1.2, min(deadzone_deg, 6.0))

    # -------------------------------------------------------------
    # TAHAP 2: Rekam Gerakan Lambat / Presisi (Membidik Tombol Kecil)
    # -------------------------------------------------------------
    slow_data = collect_for_seconds(
        4, "TAHAP 2 - Gerakkan HP ПELAN & HALUS seolah sedang membidik ikon kecil (4 detik)"
    )
    slow_speeds = [np.hypot(d.get("yawRate", 0), d.get("pitchRate", 0)) for d in slow_data if "yawRate" in d]
    precision_speed_ref = float(np.median(slow_speeds)) if slow_speeds else 12.0

    # -------------------------------------------------------------
    # TAHAP 3: Rekam Gerakan Cepat (Melintasi Ujung ke Ujung Layar)
    # -------------------------------------------------------------
    fast_data = collect_for_seconds(
        4, "TAHAP 3 - Ayunkan HP agak CEPAT ke kiri-kanan & atas-bawah melintasi layar (4 detik)"
    )
    fast_speeds = [np.hypot(d.get("yawRate", 0), d.get("pitchRate", 0)) for d in fast_data if "yawRate" in d]
    fast_speed_ref = float(np.percentile(fast_speeds, 75)) if fast_speeds else 65.0

    # -------------------------------------------------------------
    # TAHAP 4: Kalibrasi Arah Sumbu & Sudut Nyaman Tilt-Scroll
    # -------------------------------------------------------------
    Right_data = collect_for_seconds(
        3, "TAHAP 4 - Arahkan ujung HP ke KANAN lalu balik ke tengah berulang kali (3 detik)"
    )
    # Cek apakah sumbu horizontal dominan di yawRate (gamma) atau rollRate (beta)
    avg_abs_yaw  = np.mean([abs(d.get("yawRate", 0)) for d in Right_data]) if Right_data else 1.0
    avg_abs_roll = np.mean([abs(d.get("rollRate", 0)) for d in Right_data]) if Right_data else 0.0
    horiz_axis = "yawRate" if avg_abs_yaw >= avg_abs_roll else "rollRate"

    # Latih Classifier AI ringan (k-NN) untuk membedakan 3 rezim gerak:
    # 0 = JITTER/DIAM, 1 = PRESISI (Lambat), 2 = SWEEP (Cepat)
    X_train, y_train = [], []
    for s in still_speeds:
        X_train.append([s])
        y_train.append(0)
    for s in slow_speeds:
        X_train.append([s])
        y_train.append(1)
    for s in fast_speeds:
        X_train.append([s])
        y_train.append(2)

    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)

    profile = {
        "deadzone_deg": deadzone_deg,
        "precision_speed_ref": precision_speed_ref,
        "fast_speed_ref": fast_speed_ref,
        "horiz_axis": horiz_axis,
        "invert_x": -1.0,
        "invert_y": -1.0,
        "scroll_deadzone_deg": 3.0,
        "motion_classifier": clf
    }

    joblib.dump(profile, PROFILE_FILE)
    client.loop_stop()

    print("\n=========================================================")
    print(" KALIBRASI & TRAINING SELESAI! PROFIL TERSIMPAN:")
    print(f"   - File Profil         : {PROFILE_FILE}")
    print(f"   - Deadzone Getaran    : {deadzone_deg:.2f} deg/s")
    print(f"   - Kecepatan Presisi   : {precision_speed_ref:.2f} deg/s")
    print(f"   - Kecepatan Cepat     : {fast_speed_ref:.2f} deg/s")
    print(f"   - Sumbu Horizontal HP : {horiz_axis}")
    print("=========================================================")
    print("Sekarang jalankan: python laptop_receiver.py")

if __name__ == "__main__":
    main()