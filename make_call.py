#!/usr/bin/env python3
import os
import sys
import argparse
from modules.twilio_utils import make_call

def main():
    parser = argparse.ArgumentParser(description="Trigger Twilio outbound call (CLI)")
    parser.add_argument("to", help="Destination phone number (E.164), e.g. +201234567890")
    parser.add_argument("--base-url", help="Public base URL (ngrok or deployed) e.g. https://abcd.ngrok.io")
    parser.add_argument("--from-number", help="Optional Twilio From number")
    args = parser.parse_args()

    base = args.base_url or os.getenv("BASE_URL")
    if not base:
        print("Error: base URL not set. Provide --base-url or set BASE_URL env var.")
        sys.exit(1)

    try:
        r = make_call(to=args.to, base_url=base, from_number=args.from_number)
        print("Call created:", r)
    except Exception as e:
        print("Failed to create call:", e)
        sys.exit(2)

if __name__ == "__main__":
    main()