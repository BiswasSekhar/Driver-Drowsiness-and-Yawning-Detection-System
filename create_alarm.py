import numpy as np
from scipy.io.wavfile import write

# Generate a simple alarm sound
def generate_alarm_sound():
    # Parameters
    sample_rate = 44100  # Sample rate in Hz
    duration = 1.0       # Duration in seconds
    frequency = 440.0    # Frequency in Hz (A4 note)
    
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Generate sine wave (alarm tone)
    tone = np.sin(2 * np.pi * frequency * t)
    
    # Generate a more complex alarm by adding another frequency
    tone2 = np.sin(2 * np.pi * (frequency * 1.5) * t)
    
    # Combine tones
    alarm_tone = (tone + tone2) / 2.0
    
    # Normalize to 16-bit range
    alarm_tone = alarm_tone * 32767 / np.max(np.abs(alarm_tone))
    alarm_tone = alarm_tone.astype(np.int16)
    
    # Write to WAV file
    write('sounds/alarm.wav', sample_rate, alarm_tone)
    print("Alarm sound created at sounds/alarm.wav")

if __name__ == "__main__":
    import os
    # Create sounds directory if it doesn't exist
    os.makedirs('sounds', exist_ok=True)
    generate_alarm_sound()