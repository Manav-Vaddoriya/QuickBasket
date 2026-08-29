import time
import random

CROCKFORD_BASE32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

def encode_base32(value, length):
    """
    Encode an integer to Crockford's Base32 string of a fixed length.
    """
    chars = []
    for _ in range(length):
        chars.append(CROCKFORD_BASE32[value % 32])
        value //= 32
    return "".join(reversed(chars))

def new_ulid(timestamp_ms=None):
    """
    Generate a 26-character ULID string.
    If timestamp_ms is provided (as a float or integer millisecond timestamp), 
    it is used for the first 48 bits (10 characters) of the ULID.
    Otherwise, the current time is used.
    The remaining 80 bits (16 characters) are generated randomly.
    """
    if timestamp_ms is None:
        timestamp_ms = int(time.time() * 1000)
    else:
        timestamp_ms = int(timestamp_ms)
        
    # 48-bit timestamp
    ts_val = timestamp_ms & 0xFFFFFFFFFFFF
    ts_encoded = encode_base32(ts_val, 10)
    
    # 80-bit randomness
    rand_val = random.getrandbits(80)
    rand_encoded = encode_base32(rand_val, 16)
    
    return ts_encoded + rand_encoded
