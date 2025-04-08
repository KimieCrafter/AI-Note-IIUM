import serial
import struct

# Replace with your actual COM port
PORT = 'COM3'
BAUDRATE = 115200

ser = serial.Serial(PORT, BAUDRATE)

def angle_to_direction(angle):
    """Convert angle to cardinal direction."""
    # Normalize angle to be within 0-359 degrees
    angle = angle % 360

    if 0 <= angle < 45 or 315 <= angle < 360:
        return "North"
    elif 45 <= angle < 135:
        return "East"
    elif 135 <= angle < 225:
        return "South"
    elif 225 <= angle < 315:
        return "West"
    else:
        return "Unknown"  # This should ideally never be reached

def parse_packet(packet):
    if len(packet) != 22 or packet[0] != 0xFA:
        return None

    index = packet[1] - 0xA0
    measurements = []

    for i in range(4):
        offset = 4 + i * 4
        # Distance: 2 bytes (little endian)
        dist_l = packet[offset]
        dist_h = packet[offset + 1]
        distance = (dist_h << 8) | dist_l
        # Signal strength: 1 byte
        strength = packet[offset + 2]

        angle = (index * 4 + i) % 360  # Ensure angle is within 0-359°
        direction = angle_to_direction(angle)
        measurements.append({
            'angle': angle,
            'direction': direction,
            'distance': distance / 1000.0,  # convert to meters
            'strength': strength
        })

    return measurements

print("[*] Reading LDS-01 data...")
buffer = bytearray()

try:
    while True:
        # Continuously fill buffer
        buffer += ser.read(1)

        # Process packets
        while len(buffer) >= 22:
            if buffer[0] == 0xFA:
                packet = buffer[:22]
                data = parse_packet(packet)
                if data:
                    for point in data:
                        print(f"Angle: {point['angle']}°, Direction: {point['direction']}, Distance: {point['distance']} m, Strength: {point['strength']}")
                buffer = buffer[22:]
            else:
                # Discard until we hit 0xFA
                buffer = buffer[1:]
except KeyboardInterrupt:
    print("\n[!] Stopped by user.")
finally:
    ser.close()
