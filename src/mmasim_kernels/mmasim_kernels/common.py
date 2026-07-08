def encode_fp4(x: float) -> int:
    if x.hex() == "-0x0.0p+0":  # -0.0
        return 0b1000
    encoding = {
        0.0: 0b0000,
        0.5: 0b0001,
        1.0: 0b0010,
        1.5: 0b0011,
        2.0: 0b0100,
        3.0: 0b0101,
        4.0: 0b0110,
        6.0: 0b0111,
    }
    if abs(x) in encoding:
        return encoding[abs(x)] | (0b1000 if x < 0 else 0)
    else:
        raise ValueError(f"Unsupported value for fp4 encoding: {x}")

