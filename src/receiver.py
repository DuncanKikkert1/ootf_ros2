# =============================================================================
# Name        : receiver.py
# Author      : Duncan Kikkert
# Created     : 13/4/2026
# Last Update : 14/4/2026
# Version     : 1.0
# Description : TCP server that receives and validates messages from a TCP client. 
#               Parses valid messages into named variables according to 
#               standard Affix format.
#
# Message format : key_cmd;status;extra_num;gripper;x;y;z;aw;ap;ar
# Example        : 250;254;0;0;1.57;0;1.57;1.57;-1.57;0
#
# Dependencies : socket, re (both built-in, no install required)
# Usage        : Run this script on the Ubuntu PC, then connect TCP client
#                using this machine's IP (192.168.1.79) and the port below (9000).
# =============================================================================

import socket
import re
import json

HOST = '0.0.0.0'
PORT = 9000
FORWARD_HOST = '127.0.0.1'
FORWARD_PORT = 9001

# -----------------------------------------------------------------------------
# Validation pattern
# Expects 10 semicolon-separated fields:
#   Field 1       : one or more digits (command code)
#   Fields 2-10   : optional sign (+ or -) followed by digits, optional decimal
# -----------------------------------------------------------------------------
PATTERN = re.compile(
    r'^\d+;'
    r'\d+;'
    r'\d+;'
    r'[+\-]?\d+(\.\d+)?;'
    r'[+\-]?\d+(\.\d+)?;'
    r'[+\-]?\d+(\.\d+)?;'
    r'[+\-]?\d+(\.\d+)?;'
    r'[+\-]?\d+(\.\d+)?;'
    r'[+\-]?\d+(\.\d+)?;'
    r'[+\-]?\d+(\.\d+)?$'
)

# -----------------------------------------------------------------------------
# parse_message()
# Splits a validated message string into named variables and returns them
# as a dictionary. Command codes are stored as int, coordinates as float.
# -----------------------------------------------------------------------------
def parse_message(message):
    fields = message.split(';')
    parsed = {
        'key_cmd':   int(fields[0]),
        'status':    int(fields[1]),
        'extra_num': int(fields[2]),
        'gripper':   float(fields[3]),
        'x':         float(fields[4]),
        'y':         float(fields[5]),
        'z':         float(fields[6]),
        'aw':        float(fields[7]),
        'ap':        float(fields[8]),
        'ar':        float(fields[9]),
    }
    return parsed


# -----------------------------------------------------------------------------
# forward_message()
# Opens a short-lived connection to ros_publisher.py and sends the parsed
# message as a JSON-encoded string so ros_publisher.py can access named fields.
# -----------------------------------------------------------------------------
def forward_message(parsed):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as fwd:
            fwd.connect((FORWARD_HOST, FORWARD_PORT))
            fwd.sendall(json.dumps(parsed).encode('utf-8'))
            print(f"Forwarded to ROS publisher: {parsed}")
    except ConnectionRefusedError:
        print("ROS publisher not reachable — is ros_publisher.py running?")
    except Exception as e:
        print(f"Forwarding failed: {e}")
# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(1)
    print(f"Listening on port {PORT}...")

    try:
        conn, addr = s.accept()
        with conn:
            print(f"Welcome {addr}, please have a drink and stay a while...")
            while True:
                data = conn.recv(1024)
                if not data:
                    print("Client broke connection, closing connection...")
                    break

                message = data.decode('utf-8').strip()

                if PATTERN.match(message):
                    print(f"Message received: {message}")
                    parsed = parse_message(message=message)
                    print(f"Valid message received: {message}")
                    print(f"  key_cmd   : {parsed['key_cmd']}")
                    print(f"  status    : {parsed['status']}")
                    print(f"  extra_num : {parsed['extra_num']}")
                    print(f"  gripper   : {parsed['gripper']}")
                    print(f"  x         : {parsed['x']} mm")
                    print(f"  y         : {parsed['y']} mm")
                    print(f"  z         : {parsed['z']} mm")
                    print(f"  aw        : {parsed['aw']} deg")
                    print(f"  ap        : {parsed['ap']} deg")
                    print(f"  ar        : {parsed['ar']} deg")
                    forward_message(parsed=parsed)
                else:
                    fields = message.split(';')
                    if len(fields) != 10:
                        print(f"Invalid message rejected: expected 10 fields, got {len(fields)}: {message}")
                    else:
                        for name, value in zip(
                            ['key_cmd', 'status', 'extra_num', 'gripper', 'x', 'y', 'z', 'aw', 'ap', 'ar'],
                            fields
                        ):
                            if not re.fullmatch(r'\d+' if name in ('key_cmd', 'status', 'extra_num') else r'[+\-]?\d+(\.\d+)?', value):
                                print(f"Invalid message rejected: field '{name}' has invalid value '{value}'")
                    
    except KeyboardInterrupt:
        print("\nShutting down cleanly...")
            